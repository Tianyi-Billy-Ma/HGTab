import logging


logger = logging.getLogger(__name__)

from pprint import pprint
from easydict import EasyDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


from transformers import AutoConfig, AutoModel

from .base_executor import BaseExecutor


class Llama3Executor(BaseExecutor):
    def __init__(self, config, data_loader):
        super(Llama3Executor, self).__init__(config, data_loader)

        self.tokenizer = data_loader.tokenizer
        self.decoder_tokenizer = data_loader.decoder_tokenizer

        ModelClass = globals()[self.config.model_config.ModelClass]
        ConfigClass = globals()[self.config.model_config.ConfigClass]
        model_config = ConfigClass.from_pretrained(
            self.config.model_config.ConfigModelVersion
        )
        pprint(model_config)

        if self.config.model_config.pretrained:
            self.model = ModelClass.from_pretrained(
                self.config.model_config.ModelVersion, config=model_config
            ).half()
        else:
            self.model = ModelClass(model_config, config=model_config)

        self.model.resize_token_embeddings(len(self.tokenizer))

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
        self.optimizer = torch.optim.Adam(
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
        # TODO: Implement training step
        return None

    def validation_step(self, sample_batched, batch_idx, dataloader_idx=0):
        return self._prediction_step(sample_batched, batch_idx)

    def test_step(self, sample_batched, batch_idx, dataloader_idx=0):
        return self._prediction_step(sample_batched, batch_idx)

    def _prediction_step(self, sample_batched, batch_idx):
        """
        Perform the prediction step
        """

        predictions = []
        table_entries = []

        test_batch = EasyDict()

        outputs = self.model(**test_batch)

        return None
