// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.

// SPDX-License-Identifier: CC-BY-NC-4.0

local base_env = import '../base_env.jsonnet';

local train_batch_size = 32;
local valid_batch_size = 32;
local test_batch_size = 32;
local valid_step_size = 200;
local save_interval = 200;
local break_interval = 3000;
local train_epochs = 9999;
local adam_epsilon = 1e-08;
local lr = 1e-4;
local gradient_accumulation_steps = 4;
local gradient_clipping = 0;
local warmup_steps = 0;

local seed = 2022;


local override = {
  platform_type: 'pytorch',
  ignore_pretrained_weights: [],
  experiment_name: 'default_test',
  seed: seed,
  model_config: {
    base_model: 'Contriever',
    ModelClass: 'Contriever',
    QueryEncoderModelClass: 'DPRQuestionEncoder',
    QueryEncoderConfigClass: 'DPRConfig',
    QueryEncoderModelVersion: 'facebook/contriever-msmarco',
    ItemEncoderModelClass: 'DPRContextEncoder',
    ItemEncoderConfigClass: 'DPRConfig',
    ItemEncoderModelVersion: 'facebook/dpr-ctx_encoder-single-nq-base',
    TokenizerClass: 'DPRQuestionEncoderTokenizer',
    TokenizerModelVersion: 'facebook/dpr-question_encoder-single-nq-base',
    DecoderTokenizerClass: 'DPRContextEncoderTokenizer',
    DecoderTokenizerModelVersion: 'facebook/dpr-ctx_encoder-single-nq-base',
    pretrained: 1,
    modules: [
      'separate_query_and_item_encoders',
    ],
    Ks: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 50, 80, 100],
    num_negative_samples: 4,
    prepend_tokens: {
      query_encoder: '',
      item_encoder: '',
    },
    SPECIAL_TOKENS: {
      additional_special_tokens: [],
    },
    DECODER_SPECIAL_TOKENS: {
      additional_special_tokens: [
        '<HEADER>',
        '<HEADER_SEP>',
        '<HEADER_END>',
        '<ROW>',
        '<ROW_SEP>',
        '<ROW_END>',
        '<COL>',
        '<COL_SEP>',
        '<COL_END>',
        '<CELL>',
        '<CELL_END>',
      ],
    },
    input_modules: {
      module_list: [
        {
          type: 'QuestionInput',
          option: 'default',
          separation_tokens: { start: '', end: '' },
        },
      ],
      postprocess_module_list: [
        { type: 'PostProcessInputTokenization', option: 'default' },
      ],
    },
    decoder_input_modules: {
      module_list: [
        // {
        //   type: 'TextBasedTableInput',
        //   option: 'default',
        //   separation_tokens: { header_start: '<HEADER>', header_sep: '<HEADER_SEP>', header_end: '<HEADER_END>', row_start: '<ROW>', row_sep: '<ROW_SEP>', row_end: '<ROW_END>' },
        // },
        {
          type: 'HypergraphInput',
          option: 'default',
          separation_tokens: {
            header_start: '<HEADER>',
            header_sep: '<HEADER_SEP>',
            header_end: '<HEADER_END>',
            row_start: '<ROW>',
            row_sep: '<ROW_SEP>',
            row_end: '<ROW_END>',
            col_start: '<COL>',
            col_sep: '<COL_SEP>',
            col_end: '<COL_END>',
            cell_start: '<CELL>',
            cell_end: '<CELL_END>',
          },
        },
      ],
      postprocess_module_list: [
        { type: 'PostProcessDecoderInputHGTokenization', option: 'default' },
        { type: 'PostProcessHG', option: 'default' },
      ],
    },
    output_modules: {
      module_list: [
        { type: 'SimilarityOutput', option: 'default' },
      ],
      postprocess_module_list: [
        { type: 'PostProcessConcatenateLabels', option: 'default' },
      ],
    },
  },
  data_loader: {
    type: 'DataLoaderForTableQA',
    dummy_dataloader: 0,
    additional: {
      max_source_length: 512,
      max_decoder_source_length: 512,
      max_target_length: 128,
    },
    dataset_modules: {
      module_list: [
        'LoadWikiTQData',
        'LoadDataLoaders',
      ],
      module_dict: {
        LoadWikiTQData: {
          type: 'LoadWikiTQData',
          option: 'default',
          config: {
            preprocess: ['create_table_with_neg_samples'],
            path: {
              train: 'TableQA_data/wikitq/preprocessed_create_table_with_neg_samples_split_table_train.arrow',
              validation: 'TableQA_data/wikitq/preprocessed_create_table_with_neg_samples_split_table_validation.arrow',
              test: 'TableQA_data/wikitq/preprocessed_create_table_with_neg_samples_split_table_test.arrow',
            },
          },
        },
        LoadDataLoaders: {
          type: 'LoadDataLoaders',
          option: 'default',
          config: {
            train: [
              {
                dataset_type: 'DPRRAGWikiTQDataset',
                split: 'train',
                use_column: 'wtq_data',
              },
            ],
            valid: [
              {
                dataset_type: 'DPRRAGWikiTQDataset',
                split: 'validation',
                use_column: 'wtq_data',
              },
              {
                dataset_type: 'DPRRAGWikiTQDataset',
                split: 'test',
                use_column: 'wtq_data',
              },
            ],
            test: [
              {
                dataset_type: 'DPRRAGWikiTQDataset',
                split: 'train',
                use_column: 'wtq_data',
              },
              {
                dataset_type: 'DPRRAGWikiTQDataset',
                split: 'validation',
                use_column: 'wtq_data',
              },
              {
                dataset_type: 'DPRRAGWikiTQDataset',
                split: 'test',
                use_column: 'wtq_data',
              },
            ],
          },
        },
      },
    },
  },
  // cuda: 0,
  // gpu_device: 0,
  train: {
    type: 'HGExecutor',
    epochs: train_epochs,
    batch_size: train_batch_size,
    lr: lr,
    adam_epsilon: adam_epsilon,
    load_epoch: -1,
    load_model_path: '',
    load_best_model: 0,
    save_interval: save_interval,
    scheduler: 'none',
    additional: {
      gradient_accumulation_steps: gradient_accumulation_steps,
      warmup_steps: warmup_steps,
      gradient_clipping: gradient_clipping,
      save_top_k_metric: 'valid/DPRRAGWikiTQDataset.validation/recall_at_5',
    },
  },
  valid: {
    batch_size: valid_batch_size,
    step_size: valid_step_size,
    break_interval: break_interval,
    additional: {
    },
  },
  test: {
    evaluation_name: 'test_evaluation',
    load_epoch: -1,
    load_model_path: '',
    load_best_model: 0,
    batch_size: test_batch_size,
    num_evaluation: 0,
    additional: {
      multiprocessing: 4,
    },
  },
  metrics: [
    { name: 'compute_TQA_retrieval_results' },
  ],
};

std.mergePatch(base_env, override)
