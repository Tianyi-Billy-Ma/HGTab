import math
import time
import os
import sys
import scipy
import datetime
import numpy as np
import json
import operator
import pickle
from trainers.base_executor import BaseExecutor
import wandb
import logging

logger = logging.getLogger(__name__)

from pprint import pprint
from tqdm import tqdm
from easydict import EasyDict
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

import pytorch_lightning as pl
from pytorch_lightning import Trainer, seed_everything

# For TAPEX model
from transformers import TapexTokenizer, BartConfig, BartForConditionalGeneration
from transformers.models.auto.modeling_auto import (
    MODEL_FOR_CAUSAL_LM_MAPPING_NAMES,
    MODEL_MAPPING_NAMES,
)
from transformers.modeling_utils import unwrap_model

from .metrics_processors import MetricsProcessor
from .base_executor import BaseExecutor
from utils.dirs import *
from models.dpr.dpr_retriever import RetrieverDPR
from models.allset.dpr_allset import RetrieverDPRHG


class HGExecutor(BaseExecutor):
    def __init__(self, config, data_loader):
        super().__init__(config, data_loader)

        self.tokenizer = data_loader.tokenizer
        self.decoder_tokenizer = data_loader.decoder_tokenizer

        ModelClass = globals()[self.config.model_config.ModelClass]

        self.model = ModelClass(config=config)
        self.model.resize_token_embeddings(
            len(self.tokenizer), len(self.decoder_tokenizer)
        )

    def configure_optimizers(self):
        """
        Return optimizers and schedulers
        """

        def get_parameter_names(model, forbidden_layer_types):
            """
            Returns the names of the model parameters that are not inside a forbidden layer.
            """
            result = []
            for name, child in model.named_children():
                result += [
                    f"{name}.{n}"
                    for n in get_parameter_names(child, forbidden_layer_types)
                    if not isinstance(child, tuple(forbidden_layer_types))
                ]
            # Add model specific parameters (defined with nn.Parameter) since they are not in any child.
            result += list(model._parameters.keys())
            return result

        weight_decay = self.config.train.additional.get("weight_decay", 0)
        if weight_decay == 0:
            optimization_parameters = [
                {
                    "params": [p for n, p in self.model.named_parameters()],
                    "lr": self.config.train.lr,
                    "initial_lr": self.config.train.lr,
                },
            ]
        else:
            # The weight decay to apply (if not zero) to all layers except all bias and LayerNorm weights in [`AdamW`]
            ALL_LAYERNORM_LAYERS = [nn.LayerNorm]

            decay_parameters = get_parameter_names(self.model, ALL_LAYERNORM_LAYERS)
            decay_parameters = [name for name in decay_parameters if "bias" not in name]
            optimization_parameters = [
                {
                    "params": [
                        p
                        for n, p in self.model.named_parameters()
                        if n in decay_parameters
                    ],
                    "weight_decay": weight_decay,
                    "lr": self.config.train.lr,
                    "initial_lr": self.config.train.lr,
                },
                {
                    "params": [
                        p
                        for n, p in self.model.named_parameters()
                        if n not in decay_parameters
                    ],
                    "weight_decay": 0.0,
                    "lr": self.config.train.lr,
                    "initial_lr": self.config.train.lr,
                },
            ]

        for group in optimization_parameters:
            logger.info(
                "#params: {}   lr: {}".format(len(group["params"]), group["lr"])
            )

        """define optimizer"""
        self.optimizer = torch.optim.AdamW(
            optimization_parameters, lr=self.config.train.lr
        )

        if self.config.train.scheduler == "linear":
            from transformers import get_linear_schedule_with_warmup

            # Using Linear scheduler
            self.scheduler = get_linear_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=self.config.train.additional.warmup_steps,
                num_training_steps=self.trainer.estimated_stepping_batches,
                last_epoch=self.global_step,
            )
        elif self.config.train.scheduler == "cosine":
            t_total = self.config.train.epochs
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, t_total, eta_min=1e-5, last_epoch=-1, verbose=False
            )
        else:
            from transformers import get_constant_schedule_with_warmup

            # Using constant scheduler
            self.scheduler = get_constant_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=self.config.train.additional.warmup_steps,
                last_epoch=self.global_step,
            )

        return {
            "optimizer": self.optimizer,
            "lr_scheduler": {
                # REQUIRED: The scheduler instance
                "scheduler": self.scheduler,
                # The unit of the scheduler's step size, could also be 'step'.
                # 'epoch' updates the scheduler on epoch end whereas 'step'
                # updates it after a optimizer update.
                "interval": "step",
                # How many epochs/steps should pass between calls to
                # `scheduler.step()`. 1 corresponds to updating the learning
                # rate after every epoch/step.
                "frequency": 1,
                # If using the `LearningRateMonitor` callback to monitor the
                # learning rate progress, this keyword can be used to specify
                # a custom logged name
                "name": None,
            },
        }

    def training_step(self, sample_batched, batch_idx):
        train_batch = {
            "input_ids": sample_batched["input_ids"].to(self.device),
            "attention_mask": sample_batched["attention_mask"].to(self.device),
            "labels": sample_batched["labels"].to(self.device),
            "item_input_ids": sample_batched["decoder_input_ids"].to(self.device),
            "item_attention_mask": sample_batched["decoder_input_attention_mask"].to(
                self.device
            ),
        }

        forward_results = self.model(**train_batch)
        batch_loss = forward_results.loss

        # if unwrap_model(self.model)._get_name() in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
        #     batch_loss = self.label_smoother(forward_results, train_batch.labels, shift_labels=True)
        # else:
        #     batch_loss = self.label_smoother(forward_results, train_batch.labels)

        # log the current learning rate from shedulers
        current_lrs = self.scheduler.get_last_lr()
        for index, current_lr in enumerate(current_lrs):
            self.log(
                f"train/lr[{index}]",
                current_lr,
                prog_bar=True,
                on_step=True,
                logger=True,
                sync_dist=True,
            )

        # logs metrics for each training_step,
        # and the average across the epoch, to the progress bar and logger
        self.log(
            "train/loss",
            batch_loss,
            on_step=True,
            on_epoch=True,
            logger=True,
            sync_dist=True,
        )

        data_to_return = {
            "loss": batch_loss,
        }
        return data_to_return

    def on_validation_epoch_start(self):
        self.validation_step_outputs = [[]] * len(self.val_dataloader())

    def validation_step(self, sample_batched, batch_idx, dataloader_idx=0):
        return self._compute_query_embeddings_step(sample_batched, batch_idx)

    def on_validation_batch_end(self, outputs, batch, batch_idx, dataloader_idx=0):
        self.validation_step_outputs[dataloader_idx].append(outputs)

    def on_validation_epoch_end(self, validation_step_outputs=None):
        validation_step_outputs = self.validation_step_outputs
        for i in range(len(self.val_dataloader())):
            validation_step_output = validation_step_outputs[i]
            if len(validation_step_output) > 0:
                log_dict = self.evaluate_outputs(
                    validation_step_output,
                    self.val_dataloader()[i],
                    self.val_dataloader_names[i],
                )
            self.logging_results(log_dict, prefix=self.val_dataloader_names[i])
        return None

    def on_test_batch_start(self, sample_batched, batch_idx, dataloader_idx=0):
        # This is called when the test epoch starts.
        # Initialize the test_step_outputs to store the outputs of each test step.
        self.test_step_outputs = [[]] * len(self.test_dataloader())

    def test_step(self, sample_batched, batch_idx, dataloader_idx=0):
        return self._compute_embeddings_step(sample_batched, batch_idx)

    def on_test_batch_end(self, outputs, batch, batch_idx, dataloader_idx=0):
        self.test_step_outputs[dataloader_idx].append(outputs)

    def on_test_epoch_end(self):
        test_step_outputs = self.test_step_outputs
        self.save_HF_model()
        for i in range(len(self.test_dataloader())):
            if len(self.test_dataloader()) == 1:
                test_step_output = test_step_outputs[0]
            else:
                test_step_output = test_step_outputs[i]
            if len(test_step_output) > 0:
                log_dict = self.evaluate_outputs(
                    test_step_output,
                    self.test_dataloader()[i],
                    self.test_dataloader_names[i],
                )
                self.logging_results(
                    log_dict,
                    prefix=f"{self.config.test.evaluation_name}_{self.test_dataloader_names[i]}",
                )

        return None

    def _compute_query_embeddings_step(self, sample_batched, batch_idx):
        test_query_batch = {
            "input_ids": sample_batched["input_ids"].to(self.device),
            "attention_mask": sample_batched["attention_mask"].to(self.device),
        }
        query_emb = self.model.generate_query_embeddings(**test_query_batch)

        test_item_batch = {
            "node_input_ids": sample_batched["node_input_ids"].to(self.device),
            "node_input_attention_mask": sample_batched["node_input_attention_mask"].to(
                self.device
            ),
            "hyperedge_input_ids": sample_batched["hyperedge_input_ids"].to(
                self.device
            ),
            "hyperedge_input__attention_mask": sample_batched[
                "hyperedge_input_attention_mask"
            ].to(self.device),
            "edge_index": sample_batched["edge_index"].to(self.device),
            "table_index": sample_batched["table_index"].to(self.device),
        }
        item_emb = self.model.generate_item_embeddings(**test_item_batch)
        data_to_return = {
            "batch_idx": batch_idx,
            "query_emb": query_emb,
            "item_emb": item_emb,
            "question_ids": sample_batched["question_ids"],
            "answers": sample_batched["answers"],
        }
        return data_to_return

    def evaluate_outputs(self, outputs, dataloader, dataloader_name, mode="test"):
        # Compute the scores

        return log_dict

    def save_HF_model(self):
        if self.global_rank != 0:
            logger.info("global rank is not 0, skip saving models")
            return
        logger.info("Saving model in the Huggingface format...")
        path_save_model = os.path.join(
            self.config.saved_model_path, "step_{}".format(self.global_step)
        )
        self.model.query_encoder.save_pretrained(
            os.path.join(path_save_model, "query_encoder")
        )
        self.data_loader.tokenizer.save_pretrained(
            os.path.join(path_save_model, "query_encoder_tokenizer")
        )
        self.model.item_encoder.save_pretrained(
            os.path.join(path_save_model, "item_encoder")
        )
        self.data_loader.decoder_tokenizer.save_pretrained(
            os.path.join(path_save_model, "item_encoder_tokenizer")
        )
        self.model.hypergraph_encoder.save_pretrained(
            os.path.join(path_save_model, "hypergraph_encoder")
        )
        logger.info("Model has been saved to {}".format(path_save_model))
