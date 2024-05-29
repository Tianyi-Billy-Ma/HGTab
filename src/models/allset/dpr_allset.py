import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import pytorch_lightning as pl
from transformers import T5EncoderModel, T5Config
from transformers import DPRQuestionEncoder, DPRContextEncoder, DPRConfig
from transformers import BertModel, BertConfig
from easydict import EasyDict

from torch_scatter import scatter
from torch.nn.functional import gelu


class EmbeddingLayer(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.hidden_dropout_prob = config.hidden_dropout_prob
        self.vocab_size = config.vocab_size
        self.hidden_size = config.hidden_size

        self.embedding = nn.Embedding(self.vocab_size, self.hidden_size, padding_idx=0)
        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(self.hidden_dropout_prob)

    def resize_token_embeddings(self, vocab_size):
        old_vocab_size = self.vocab_size
        old_embeddings = self.embedding

        new_vocab_size = vocab_size

        new_embeddings = nn.Embedding(
            new_vocab_size,
            self.hidden_size,
            device=old_embeddings.weight.device,
            dtype=old_embeddings.weight.dtype,
        )

        n = min(old_vocab_size, new_vocab_size)
        new_embeddings.weight.data[:n, :] = old_embeddings.weight.data[:n, :]

        self.embedding = new_embeddings
        self.vocab_size = new_vocab_size

    def forward(self, input_ids, attention_mask=None):
        embeddings = self.embedding(input_ids)
        pooler_output = torch.div(
            torch.sum(embeddings, dim=1),
            torch.count_nonzero(input_ids, dim=1).unsqueeze(-1),
        )
        pooler_output = self.LayerNorm(pooler_output)
        pooler_output = self.dropout(pooler_output)

        return EasyDict({"pooler_output": pooler_output})


class HGLayer(pl.LightningModule):
    def __init__(self, config):
        super().__init__()

        self.dropout = config.hidden_dropout_prob
        self.hidden_size = config.hidden_size
        self.V2E = nn.Linear(self.hidden_size, self.hidden_size)
        self.E2V = nn.Linear(self.hidden_size, self.hidden_size)
        self.fuse = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.activation = globals()[config.hidden_act]

    def reset_parameters(self):
        self.fuse.reset_parameters()
        self.V2E.reset_parameters()
        self.E2V.reset_parameters()

    def forward(self, emb_V, emb_E, edge_index):
        reversed_edge_index = torch.cat(
            [edge_index[1].unsqueeze(0), edge_index[0].unsqueeze(0)], dim=0
        )

        emb_E_tem = self.V2E(emb_V)
        emb_E_tem = scatter(
            emb_E_tem[edge_index[0]], edge_index[1], dim=0, reduce="mean"
        )
        emb_E_tem = self.activation(emb_E_tem)

        emb_E = torch.cat([emb_E, emb_E_tem], dim=-1)
        emb_E = self.fuse(emb_E)
        emb_E = F.dropout(emb_E, p=self.dropout, training=self.training)

        emb_V = self.E2V(emb_E)
        emb_V = scatter(
            emb_V[reversed_edge_index[0]],
            reversed_edge_index[1],
            dim=0,
            reduce="mean",
        )
        emb_V = self.activation(emb_V)

        emb_V = F.dropout(emb_V, p=self.dropout, training=self.training)

        return emb_V, emb_E


class AllSet(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_layers = config.num_layers
        self.layers = nn.ModuleList([HGLayer(config) for _ in range(self.num_layers)])

    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()

    def forward(self, node_embeddings, hyperedge_embeddings, edge_index):
        emb_V, emb_E = node_embeddings, hyperedge_embeddings

        edge_index[1, :] -= edge_index[1, :].min()
        num_nodes, num_hyperedges = emb_V.size(0), emb_E.size(0)
        self_loop = (
            torch.LongTensor([[i, num_hyperedges + i] for i in range(num_nodes)])
            .to(edge_index.device)
            .T
        )
        edge_index = torch.cat([edge_index, self_loop], dim=1)
        emb_E = torch.cat([emb_E, emb_V], dim=0)

        for layer in self.layers:
            emb_V, emb_E = layer(emb_V, emb_E, edge_index)
        return emb_V, emb_E


class RetrieverDPRHG(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config

        QueryEncoderModelClass = globals()[
            self.config.model_config.QueryEncoderModelClass
        ]

        QueryEncoderConfigClass = globals()[
            self.config.model_config.QueryEncoderConfigClass
        ]
        query_model_config = QueryEncoderConfigClass.from_pretrained(
            self.config.model_config.QueryEncoderModelVersion
        )
        self.query_encoder = QueryEncoderModelClass.from_pretrained(
            self.config.model_config.QueryEncoderModelVersion, config=query_model_config
        )

        self.SEP_ENCODER = (
            True
            if "separate_query_and_item_encoders" in self.config.model_config.modules
            else None
        )

        if self.SEP_ENCODER:
            # ItemEncoderModelClass = globals()[
            #     self.config.model_config.ItemEncoderModelClass
            # ]
            ItemEncoderConfigClass = globals()[
                self.config.model_config.ItemEncoderConfigClass
            ]
            item_model_config = ItemEncoderConfigClass.from_pretrained(
                self.config.model_config.ItemEncoderModelVersion
            )
            # self.item_encoder = ItemEncoderModelClass.from_pretrained(
            #     self.config.model_config.ItemEncoderModelVersion,
            #     config=item_model_config,
            # )

            self.item_encoder = EmbeddingLayer(item_model_config)
        else:
            # Use the same model for query and item encoders
            item_model_config = query_model_config
            self.item_encoder = self.query_encoder

        self.query_pooler = None
        self.item_pooler = None

        hypergraph_config = EasyDict(
            vocab_size=item_model_config.vocab_size,
            num_layers=2,
            hidden_size=item_model_config.hidden_size,
            hidden_act=item_model_config.hidden_act,
            hidden_dropout_prob=item_model_config.hidden_dropout_prob,
        )
        self.hypergraph_encoder = AllSet(hypergraph_config)

        self.loss_fn = nn.CrossEntropyLoss()

    def resize_token_embeddings(self, dim, decoder_dim=None):
        self.query_encoder.resize_token_embeddings(dim)
        if "separate_query_and_item_encoders" in self.config.model_config.modules:
            self.item_encoder.resize_token_embeddings(decoder_dim)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        node_input_ids=None,
        node_input_attention_mask=None,
        hyperedge_input_ids=None,
        hyperedge_input_attention_mask=None,
        labels=None,
        edge_index=None,
        table_index=None,
        **kwargs,
    ):
        query_outputs = self.query_encoder(
            input_ids=input_ids, attention_mask=attention_mask
        )
        query_embeddings = query_outputs.pooler_output
        if self.query_pooler is not None:
            query_embeddings = self.query_pooler(query_last_hidden_states)
        query_embeddings = query_embeddings.contiguous()

        # item_outputs = self.item_encoder(input_ids=item_input_ids,
        #                                 attention_mask=item_attention_mask)
        # item_embeddings = item_outputs.pooler_output
        # if self.item_pooler is not None:
        #     item_embeddings = self.item_pooler(item_last_hidden_states)

        # query_embeddings = query_embeddings.contiguous()
        # item_embeddings = item_embeddings.contiguous()

        node_outputs = self.item_encoder(
            input_ids=node_input_ids, attention_mask=node_input_attention_mask
        )
        hyperedge_outputs = self.item_encoder(
            input_ids=hyperedge_input_ids, attention_mask=hyperedge_input_attention_mask
        )
        node_embeddings = node_outputs.pooler_output
        hyperedge_embeddings = hyperedge_outputs.pooler_output
        if self.item_pooler is not None:
            node_embeddings = self.item_pooler(node_last_hidden_states)
            hyperedge_embeddings = self.item_pooler(hyperedge_last_hidden_states)

        emb_V, emb_E = self.hypergraph_encoder(
            node_embeddings, hyperedge_embeddings, edge_index
        )
        item_embeddings = emb_E[table_index]

        batch_size = query_embeddings.shape[0]
        batch_size_with_pos_and_neg = item_embeddings.shape[0]
        num_pos_and_neg = batch_size_with_pos_and_neg // batch_size
        num_pos, num_neg = 1, num_pos_and_neg - 1

        in_batch_labels = torch.zeros(batch_size, batch_size_with_pos_and_neg).to(
            labels.device
        )
        step = num_pos_and_neg
        for i in range(batch_size):
            in_batch_labels[i, i * step] = 1
        in_batch_labels = torch.argmax(in_batch_labels, dim=1)
        in_batch_scores = torch.matmul(query_embeddings, item_embeddings.T)
        loss = self.loss_fn(in_batch_scores, in_batch_labels)
        return EasyDict({"loss": loss})

    def generate_query_embeddings(self, input_ids=None, attention_mask=None):
        query_outputs = self.query_encoder(
            input_ids=input_ids, attention_mask=attention_mask
        )
        query_last_hidden_states = query_outputs.pooler_output
        if self.query_pooler is not None:
            query_last_hidden_states = self.query_pooler(query_last_hidden_states)
        query_embeddings = query_last_hidden_states
        return query_embeddings

    def generate_item_embeddings(
        self,
        node_input_ids=None,
        node_input_attention_mask=None,
        hyperedge_input_ids=None,
        hyperedge_input_attention_mask=None,
        edge_index=None,
        table_index=None,
    ):
        node_outputs = self.item_encoder(
            input_ids=node_input_ids, attention_mask=node_input_attention_mask
        )
        hyperedge_outputs = self.item_encoder(
            input_ids=hyperedge_input_ids,
            attention_mask=hyperedge_input_attention_mask,
        )
        node_embeddings = node_outputs.pooler_output
        hyperedge_embeddings = hyperedge_outputs.pooler_output
        if self.item_pooler is not None:
            node_embeddings = self.item_pooler(node_last_hidden_states)
            hyperedge_embeddings = self.item_pooler(hyperedge_last_hidden_states)

        emb_V, emb_E = self.hypergraph_encoder(
            node_embeddings, hyperedge_embeddings, edge_index
        )
        item_embeddings = emb_E[table_index]
        return item_embeddings
