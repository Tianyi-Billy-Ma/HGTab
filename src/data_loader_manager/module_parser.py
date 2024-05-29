# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.

# SPDX-License-Identifier: CC-BY-NC-4.0

from ast import Raise
from typing import Optional
from easydict import EasyDict
import torch
import pandas as pd
import numpy as np
from pprint import pprint
import random

from utils.tapas_utils import prepare_table_in_sqa_format
from utils.hypergraphs import BipartiteData


class ModuleParser:
    """
    This is a module inherited by the dataset class
    This class is used to parse the sample to form input/output/decoder_input data
    It should be able to process both text-based features and image-based features
    Process:
        (1) Sample-level Sub Parsers:
            Add data fields to the sample
        (2) Porcessing:
            Aggregating features from individual sub parsers
            Strings under the same field will be automatically concatenated
            Use different fields for different image-based features
        (3) Post-processing:
            Add post-processing units to process the data after parsing
            e.g. tokenization, adding new sample-level features
    """

    def __init__(self) -> None:
        pass

    def HypergraphInput(self, sample: EasyDict, module: EasyDict) -> EasyDict:
        node_sentences, hyperedge_sentences = [], []
        table = sample.table
        num_rows, num_cols = len(table.rows), len(table.header)
        row_edge_index, col_edge_index = [], []
        for he_idx in range(num_rows):
            # Build row hyperedge index
            row_edge_index.extend(
                [[he_idx * num_cols + col, he_idx] for col in range(num_cols)]
            )
            # Build row hyperedge sentence
            row_sentence = [module.separation_tokens.row_start]
            for col in range(num_cols):
                row_sentence += [table.rows[he_idx][col]]
                if col < num_cols - 1:
                    row_sentence += [module.separation_tokens.row_sep]
            row_sentence += [module.separation_tokens.row_end]
            hyperedge_sentences.append(" ".join(row_sentence).strip())

        for he_idx in range(num_cols):
            # Build column hyperedge index
            col_edge_index.extend(
                [[row * num_cols + he_idx, he_idx] for row in range(num_rows)]
            )

            # Build row hyperedge sentence
            col_sentence = [module.separation_tokens.col_start]
            for row in range(num_rows):
                col_sentence += [table.rows[row][he_idx]]
                if row < num_rows - 1:
                    col_sentence += [module.separation_tokens.col_sep]
            col_sentence += [module.separation_tokens.col_end]
            hyperedge_sentences.append(" ".join(col_sentence).strip())

        for row in range(num_rows):
            for col in range(num_cols):
                node_sentence = [
                    module.separation_tokens.cell_start,
                    table.rows[row][col],
                    module.separation_tokens.cell_end,
                ]
                node_sentences.append(" ".join(node_sentence).strip())
        table_sentence = self.TextBasedTableInput(sample, module).text_sequence
        row_edge_index = np.array(row_edge_index)
        col_edge_index = np.array(col_edge_index)
        hg_edge_index = np.array(
            [[node_idx, 0] for node_idx in range(num_cols * num_rows)]
        )
        col_edge_index[:, 1] += num_rows  # shift the index of column hyperedge index
        hg_edge_index[:, 1] += num_rows + num_cols  # shift the index of table node
        edge_index = np.concatenate(
            [row_edge_index, col_edge_index, hg_edge_index], axis=0
        ).T  # (2, c_num_edges)

        return_dict = EasyDict(
            node_sentences=node_sentences,
            hyperedge_sentences=hyperedge_sentences + [table_sentence],
            num_nodes=num_cols * num_rows,
            num_hyperedges=num_cols + num_rows + 1,
            edge_index=edge_index,
        )
        return return_dict

    def QuestionInput(self, sample: EasyDict, module: EasyDict) -> EasyDict:
        """
        Parse the question input
        Simple add the question to the text sequence
        """
        return_dict = EasyDict(
            text_sequence="",
        )
        if module.option == "default":
            input_sequence = " ".join(
                [module.separation_tokens.start]
                + [sample.question]
                + [module.separation_tokens.end]
            ).strip()

        return_dict.text_sequence = input_sequence
        return return_dict

    def TAPASSpecificInput(self, sample: EasyDict, module: EasyDict) -> EasyDict:
        """
        Add data relevant for TAPAS tokenization to the sample
        """
        return_dict = EasyDict(
            answer_text=sample["answer_text"],
            float_value=sample["float_value"],
            aggregation_function=sample["aggregation_function"],
            answer_coordinates=sample["answer_coordinates"],
            valid=sample["valid"],
        )

        return return_dict

    def TextBasedVisionInput(
        self, sample: EasyDict, module: EasyDict
    ) -> Optional[EasyDict]:
        """
        Default TextBasedVisionInput module parser
        object: text-based objects, with attributes and OCR'ed texts
        caption: iamge captions
        """
        return_dict = EasyDict(
            text_sequence="",
        )

        # Input from Vision
        vision_sentences = []
        if module.option == "object":
            vision_sentences += [module.separation_tokens.start]
            for obj in sample.objects:
                attribute_max = module.get("attribute_max", 0)
                if attribute_max > 0:
                    # find suitable attributes
                    suitable_attributes = []
                    for attribute, att_score in zip(
                        obj["attributes"], obj["attribute_scores"]
                    ):
                        if (
                            att_score > module.attribute_thres
                            and len(suitable_attributes) < attribute_max
                        ):
                            suitable_attributes.append(attribute)
                    # append to the sentence
                    vision_sentences += suitable_attributes
                vision_sentences.append(obj["class"])
                vision_sentences.append(module.separation_tokens.sep)

            ocr = module.get("ocr", 0)
            if ocr > 0:
                text_annotations = sample.img_ocr
                filtered_descriptions = []
                for text_annoation in text_annotations:
                    description = text_annoation["description"].strip()
                    description = description.replace(
                        "\n", " "
                    )  # remove line switching
                    # vision_sentences += [description]
                    # print('OCR feature:', description)
                    if description not in filtered_descriptions:
                        filtered_descriptions.append(description)
                # print('OCR feature:', filtered_descriptions)
                vision_sentences += filtered_descriptions

            vision_sentences += [module.separation_tokens.end]
            return_dict.text_sequence = " ".join(vision_sentences)

        elif module.option == "caption":
            return_dict.text_sequence = " ".join(
                [module.separation_tokens.start]
                + [sample.img_caption["caption"]]
                + [module.separation_tokens.end]
            )

        return return_dict

    def GenerationOutput(
        self, sample: EasyDict, module: EasyDict
    ) -> Optional[EasyDict]:
        """
        Parse the default generation output from gold_answer
        """
        return_dict = EasyDict(
            text_sequence="",
        )
        output_sequence = sample.gold_answer
        return_dict.text_sequence = output_sequence
        return return_dict

    def KnowledgeInput(self, sample: EasyDict, module: EasyDict) -> Optional[EasyDict]:
        """
        Parse the knowledge input
        """
        return_dict = EasyDict(
            text_sequence="",
        )
        return_dict.text_sequence = " ".join(
            [module.separation_tokens.start]
            + [sample.passage_content]
            + [module.separation_tokens.end]
        )
        return return_dict

    def SimilarityOutput(
        self, sample: EasyDict, module: EasyDict
    ) -> Optional[EasyDict]:
        """
        Generate the similarity output
        """
        label = [1]
        label += [0] * self.config.model_config.num_negative_samples
        return_dict = EasyDict(
            label=label,
        )
        return return_dict

    def TableInput(self, sample: EasyDict, module: EasyDict) -> Optional[EasyDict]:
        """
        Transform table input to pd.DataFrame
        """
        example_table = sample.table
        table = pd.DataFrame.from_records(
            example_table["rows"], columns=example_table["header"]
        )
        if module.option == "swap_column":
            cols = list(table.columns)
            random.shuffle(cols)
            table = table[cols]

        return_dict = EasyDict(
            table=table,
            answers=sample.answers,
        )
        return return_dict

    def TextBasedTableInput(
        self, sample: EasyDict, module: EasyDict
    ) -> Optional[EasyDict]:
        """
        Serialise a table into a sentence
        "separation_tokens": {"header_start": "<HEADER>", "header_sep": "<HEADER_SEP>", "header_end": "<HEADER_END>", "row_start": "<ROW>", "row_sep": "<ROW_SEP>", "row_end": "<ROW_END>"}
        """
        table = sample.table
        table_sentences = []

        if module.get("add_title", False) == True:
            table_sentences.append(module.separation_tokens.title_start)
            # add table titles
            table_sentences.append(table.documentTitle)
            table_sentences.append(module.separation_tokens.title_end)

        table_sentences.append(module.separation_tokens.header_start)
        for column_name in table["header"]:
            table_sentences += [column_name, module.separation_tokens.header_sep]
        table_sentences.append(module.separation_tokens.header_end)

        for row in table["rows"]:
            table_sentences.append(module.separation_tokens.row_start)
            for cell in row:
                table_sentences += [cell, module.separation_tokens.row_sep]
            table_sentences.append(module.separation_tokens.row_end)

        table_sentence = " ".join(table_sentences).strip()

        return_dict = EasyDict(
            text_sequence=table_sentence,
        )
        return return_dict

    def FlattenedAnswerOutput(
        self, sample: EasyDict, module: EasyDict
    ) -> Optional[EasyDict]:
        """
        Generate the flattened answer
        Serialise answers, separated by comma
        """
        return_dict = EasyDict(
            text_sequence=", ".join(sample.answers),
            answers=sample.answers,
        )
        return return_dict

    def parse_modules(
        self,
        sample: EasyDict,
        modules: EasyDict,
        type: str,
        process_modules: Optional[EasyDict] = None,
    ) -> Optional[EasyDict]:
        """
        Parse the sample to form input/output/decoder_input
        Args:
            sample: sample to be parsed
            modules: modules to be parsed
            type: type of the module
        Returns:
            parsed sample
        """
        data_collection = []
        if type == "input":
            for input_module in modules:
                parser_func = getattr(self, input_module.type)
                parsed_data = parser_func(sample, input_module)
                data_collection.append(parsed_data)
        elif type == "decoder_input":
            for input_module in modules:
                parser_func = getattr(self, input_module.type)
                parsed_data = parser_func(sample, input_module)
                data_collection.append(parsed_data)
        elif type == "output":
            for output_module in modules:
                parser_func = getattr(self, output_module.type)
                parsed_data = parser_func(sample, output_module)
                data_collection.append(parsed_data)
        else:
            raise ValueError("Unknown type: {}".format(type))

        # Process the sample data after aggregating from individual sub parsers
        # before returning to colln_func
        processed_data = data_collection
        if process_modules is None:
            # Run default processing unit
            processed_data = self.DefaultProcessing(processed_data)
        else:
            # Run provided processing unit
            for process_module in process_modules:
                process_func = getattr(self, process_module.type)
                processed_data = process_func(processed_data)

        return processed_data

    def DefaultProcessing(self, data_to_process: EasyDict) -> EasyDict:
        """
        Process the sample data after aggregating from individual sub parsers
        """
        processed_data = EasyDict()
        for data_entry in data_to_process:
            for key, value in data_entry.items():
                if key not in processed_data:
                    processed_data[key] = value
                else:
                    if type(value) == str:
                        # automatically concatenate strings with the same key
                        processed_data[key] += " " + value
                    else:
                        raise TypeError(
                            "Undefined processing type: {}".format(type(value))
                        )

        return processed_data

    def PostProcessTAPEXInputTokenization(self, data_to_process: EasyDict) -> EasyDict:
        """
        Post-processing for TAPEX-like input tokenization
        Use TAPEX's tokenizer to tokenize inputs
        """

        assert "text_sequence" in data_to_process.keys()
        text_sequences = data_to_process.pop("text_sequence")
        assert "table" in data_to_process.keys()
        tables = data_to_process.pop("table")
        assert "answers" in data_to_process.keys()
        answers = data_to_process["answers"]
        task_prefix = ""
        if self.mode == "train":
            encoding = self.tokenizer(
                table=tables,
                query=text_sequences,
                answer=answers,
                max_length=self.config.data_loader.additional.max_source_length,
                padding="longest",
                truncation=True,
                return_tensors="pt",
            )
        else:
            encoding = self.tokenizer(
                table=tables,
                query=text_sequences,
                max_length=self.config.data_loader.additional.max_source_length,
                padding="longest",
                truncation=True,
                return_tensors="pt",
            )

        data_to_process.update(
            {
                "input_ids": encoding.input_ids,
                "attention_mask": encoding.attention_mask,
                "input_text_sequences": text_sequences,
            }
        )
        return data_to_process

    def PostProcessTAPEXOutputTokenization(self, data_to_process: EasyDict) -> EasyDict:
        """
        Post-processing for TAPEX-like output tokenization
        Use TAPEX's tokenizer to tokenize outputs
        """
        assert "text_sequence" in data_to_process.keys()
        text_sequences = data_to_process.pop("text_sequence")
        with self.decoder_tokenizer.as_target_tokenizer():
            target_encoding = self.decoder_tokenizer(
                answer=text_sequences,
                padding="longest",
                max_length=self.config.data_loader.additional.max_target_length,
                truncation=True,
                return_tensors="pt",
            )

        labels = target_encoding.input_ids
        output_sequence_ids = target_encoding.input_ids  # For teacher force training
        output_sequence_ids = torch.LongTensor(output_sequence_ids)
        output_sequence_attention_mask = torch.LongTensor(
            target_encoding.attention_mask
        )  # For teacher force training

        # replace padding token id's of the labels by -100
        labels = [
            [
                (label if label != self.decoder_tokenizer.pad_token_id else -100)
                for label in labels_example
            ]
            for labels_example in labels
        ]

        labels = torch.LongTensor(labels)
        assert labels.shape == output_sequence_ids.shape

        data_to_process.update(
            {
                "labels": labels,
                "output_sequence_ids": output_sequence_ids,
                "output_sequence_attention_mask": output_sequence_attention_mask,
                "output_text_sequences": text_sequences,
            }
        )
        return data_to_process

    def PostProcessTAPASInputTokenization(self, data_to_process: EasyDict) -> EasyDict:
        """
        Post-processing for TAPAS-like input tokenization
        Use TAPAS's tokenizer to tokenize inputs
        """
        assert "text_sequence" in data_to_process.keys()
        text_sequences = data_to_process.pop("text_sequence")
        assert "table" in data_to_process.keys()
        tables = data_to_process.pop("table")
        assert "answers" in data_to_process.keys()
        answers = data_to_process["answers"]
        assert "answer_text" in data_to_process.keys()
        answer_text = data_to_process["answer_text"]
        assert "answer_coordinates" in data_to_process.keys()
        answer_coordinates = data_to_process["answer_coordinates"]
        assert "float_value" in data_to_process.keys()
        float_value = data_to_process["float_value"]
        assert "aggregation_function" in data_to_process.keys()
        aggregation_function = data_to_process["aggregation_function"]
        assert "valid" in data_to_process.keys()
        valid = data_to_process["valid"]

        final_encoding = EasyDict({})
        # The tokenizer supports one sample at a time, so we have to process one by one
        for i in range(len(tables)):
            if self.mode == "train":
                # for training, we tokenize with more info
                encoding = self.tokenizer(
                    table=tables[i],
                    queries=text_sequences[i],
                    answer_coordinates=answer_coordinates[i],
                    answer_text=answer_text[i],
                    max_length=self.config.data_loader.additional.max_source_length,
                    truncation=True,
                    padding="max_length",
                    return_tensors="pt",
                )
            else:
                # for testing, we don't need annotations!
                encoding = self.tokenizer(
                    table=tables[i],
                    queries=text_sequences[i],
                    max_length=self.config.data_loader.additional.max_source_length,
                    truncation=True,
                    padding="max_length",
                    return_tensors="pt",
                )
            encoding = EasyDict({key: val.squeeze(0) for key, val in encoding.items()})
            for key, val in encoding.items():
                final_encoding[key] = final_encoding.get(key, []) + [val]

        # batch lists into tensors
        for key, val in final_encoding.items():
            final_encoding[key] = torch.stack(val)

        # some float values are None
        float_answer = []
        for v in float_value:
            if v is not None:
                float_answer.append(v)
            else:
                float_answer.append(np.nan)

        float_value = torch.FloatTensor(float_answer)

        # update to data_to_process
        data_to_process.update(final_encoding)
        data_to_process.update(
            {
                "tables": tables,
                "input_text_sequences": text_sequences,
                "float_value": float_value,
                "aggregation_function": aggregation_function,
                "valid": valid,
            }
        )

        return data_to_process

    def PostProcessInputTokenization(self, data_to_process: EasyDict) -> EasyDict:
        """
        Post-processing for input tokenization
        """
        assert "text_sequence" in data_to_process.keys()
        text_sequences = data_to_process.pop("text_sequence")
        task_prefix = ""
        encoding = self.tokenizer(
            [task_prefix + sequence for sequence in text_sequences],
            padding="longest",
            max_length=self.config.data_loader.additional.max_source_length,
            truncation=True,
            return_tensors="pt",
        )
        data_to_process.update(
            {
                "input_ids": encoding.input_ids,
                "attention_mask": encoding.attention_mask,
                "input_text_sequences": text_sequences,
            }
        )
        return data_to_process

    def PostProcessDecoderInputTokenization(
        self, data_to_process: EasyDict
    ) -> EasyDict:
        """
        Post-processing for decoder input tokenization
        """
        assert "text_sequence" in data_to_process.keys()
        text_sequences = data_to_process.pop("text_sequence")
        encoding = self.decoder_tokenizer(
            [sequence for sequence in text_sequences],
            padding="longest",
            max_length=self.config.data_loader.additional.max_decoder_source_length,
            truncation=True,
            return_tensors="pt",
        )
        data_to_process.update(
            {
                "decoder_input_ids": encoding.input_ids,
                "decoder_input_attention_mask": encoding.attention_mask,
                "decoder_input_text_sequences": text_sequences,
            }
        )
        return data_to_process

    def PostProcessDecoderInputHGTokenization(
        self, data_to_process: EasyDict
    ) -> EasyDict:
        assert "node_sentences" in data_to_process.keys()
        assert "hyperedge_sentences" in data_to_process.keys()

        batched_node_sentences = data_to_process.pop("node_sentences")
        batched_hyperedge_sentences = data_to_process.pop("hyperedge_sentences")

        node_encoding = self.decoder_tokenizer(
            [
                sentence
                for node_sentences in batched_node_sentences
                for sentence in node_sentences
            ],
            padding="longest",
            max_length=self.config.data_loader.additional.max_decoder_source_length,
            truncation=True,
            return_tensors="pt",
        )

        node_input_ids = node_encoding.input_ids
        node_input_attention_mask = node_encoding.attention_mask

        hyperedge_encoding = self.decoder_tokenizer(
            [
                sentence
                for hyperedge_sentences in batched_hyperedge_sentences
                for sentence in hyperedge_sentences
            ],
            padding="longest",
            max_length=self.config.data_loader.additional.max_decoder_source_length,
            truncation=True,
            return_tensors="pt",
        )

        hyperedge_input_ids = hyperedge_encoding.input_ids
        hyperedge_input_attention_mask = hyperedge_encoding.attention_mask

        data_to_process.update(
            {
                "node_input_ids": node_input_ids,
                "hyperedge_input_ids": hyperedge_input_ids,
                "node_input_attention_mask": node_input_attention_mask,
                "hyperedge_input_attention_mask": hyperedge_input_attention_mask,
            }
        )
        return data_to_process

    def PostProcessHG(self, data_to_process: EasyDict) -> EasyDict:
        assert "edge_index" in data_to_process.keys()

        batched_edge_index = data_to_process.pop("edge_index")
        batched_num_nodes = data_to_process.pop("num_nodes")
        batched_num_hyperedges = data_to_process.pop("num_hyperedges")
        num_nodes = sum(batched_num_nodes)
        num_hyperedges = sum(batched_num_hyperedges)

        edge_index = []
        for idx, edge_index_i in enumerate(batched_edge_index):
            edge_index_i[0, :] += sum(batched_num_nodes[:idx])
            edge_index_i[1, :] += sum(batched_num_hyperedges[:idx])
            edge_index.append(edge_index_i)
        edge_index = np.concatenate(edge_index, axis=1)
        assert edge_index[0].max() + 1 == num_nodes, "Node size mismatch"
        assert edge_index[1].max() + 1 == num_hyperedges, "Hyperedge size mismatch"
        table_index = [
            sum(batched_num_hyperedges[: idx + 1]) - 1
            for idx in range(len(batched_num_hyperedges))
        ]

        edge_index = torch.LongTensor(edge_index)
        table_index = torch.LongTensor(table_index)
        data_to_process.update(
            edge_index=edge_index,
            table_index=table_index,
        )
        return data_to_process

    def PostProcessOutputTokenization(self, data_to_process: EasyDict) -> EasyDict:
        """
        Post-processing for output tokenization
        """
        assert "text_sequence" in data_to_process.keys()
        text_sequences = data_to_process.pop("text_sequence")
        target_encoding = self.decoder_tokenizer(
            text_sequences,
            padding="longest",
            max_length=self.config.data_loader.additional.max_target_length,
            truncation=True,
        )
        labels = target_encoding.input_ids
        output_sequence_ids = target_encoding.input_ids  # For teacher force training
        output_sequence_ids = torch.LongTensor(output_sequence_ids)
        output_sequence_attention_mask = torch.LongTensor(
            target_encoding.attention_mask
        )  # For teacher force training

        # replace padding token id's of the labels by -100
        labels = [
            [
                (label if label != self.decoder_tokenizer.pad_token_id else -100)
                for label in labels_example
            ]
            for labels_example in labels
        ]

        labels = torch.LongTensor(labels)
        assert labels.shape == output_sequence_ids.shape

        data_to_process.update(
            {
                "labels": labels,
                "output_sequence_ids": output_sequence_ids,
                "output_sequence_attention_mask": output_sequence_attention_mask,
                "output_text_sequences": text_sequences,
            }
        )
        return data_to_process

    def PostProcessConcatenateLabels(self, data_to_process: EasyDict) -> EasyDict:
        """
        Post-processing for concatenating labels
        """
        assert "label" in data_to_process.keys()
        label = data_to_process.pop("label")
        labels = []
        for l in label:
            labels += l
        data_to_process.update(
            {
                "labels": torch.LongTensor(labels),
            }
        )
        return data_to_process

    def PostProcessColBERTQuestionInputTokenization(
        self, data_to_process: EasyDict
    ) -> EasyDict:
        """
        Post-processing for input tokenization
        """
        assert "text_sequence" in data_to_process.keys()
        text_sequences = data_to_process.pop("text_sequence")
        task_prefix = ""
        self.tokenizer.query_maxlen = (
            self.config.data_loader.additional.max_source_length
        )
        Q_ids, Q_mask = self.tokenizer.tensorize(
            [task_prefix + sequence for sequence in text_sequences]
        )
        data_to_process.update(
            {
                "input_ids": Q_ids,
                "attention_mask": Q_mask,
                "input_text_sequences": text_sequences,
            }
        )
        return data_to_process

    def PostProcessColBERTItemInputTokenization(
        self, data_to_process: EasyDict
    ) -> EasyDict:
        """
        Post-processing for decoder input tokenization
        """
        assert "text_sequence" in data_to_process.keys()
        text_sequences = data_to_process.pop("text_sequence")
        self.decoder_tokenizer.doc_maxlen = (
            self.config.data_loader.additional.max_decoder_source_length
        )
        D_ids, D_mask = self.decoder_tokenizer.tensorize(text_sequences)
        data_to_process.update(
            {
                "decoder_input_ids": D_ids,
                "decoder_input_attention_mask": D_mask,
                "decoder_input_text_sequences": text_sequences,
            }
        )
        return data_to_process

    def post_processing(
        self,
        processed_batch_data: EasyDict,
        postprocess_modules: Optional[EasyDict] = None,
        type: Optional[str] = "default",
    ) -> EasyDict:
        """
        Post-processing the processed data of the whole batch
        Called by colln_func after processing each sample
        """
        postprocessed_batch_data = processed_batch_data
        if postprocess_modules is None:
            # Do nothing and return
            return postprocessed_batch_data
        else:
            # Run provided post-processing unit
            for postprocess_module in postprocess_modules:
                process_func = getattr(self, postprocess_module.type)
                postprocessed_batch_data = process_func(postprocessed_batch_data)

        return postprocessed_batch_data
