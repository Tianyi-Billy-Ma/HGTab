// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.

// SPDX-License-Identifier: CC-BY-NC-4.0

local base_env = import '../base_env.jsonnet';

local train_batch_size = 32;
local valid_batch_size = 32;
local test_batch_size = 32;
local valid_step_size = 1000;
local save_interval = 1000;
local break_interval = 3000;
local train_epochs = 9999;
local adam_epsilon = 1e-08;
local lr = 1e-4;
local gradient_accumulation_steps = 4;
local gradient_clipping = 0;
local warmup_steps = 0;

local seed = 2022;

// here we put the index file paths
local index_files = {
  index_paths: {
    train: 'DPR_InnerTableRetrieval_wikisql_with_in_batch_neg_sampling_mixed/test/original_sets/step_11604/test.ITRWikiSQLDataset.train',
    validation: 'DPR_InnerTableRetrieval_wikisql_with_in_batch_neg_sampling_mixed/test/original_sets/step_11604/test.ITRWikiSQLDataset.validation',
    test: 'DPR_InnerTableRetrieval_wikisql_with_in_batch_neg_sampling_mixed/test/original_sets/step_11604/test.ITRWikiSQLDataset.test',
  },
};

local override = {
  platform_type: 'pytorch',
  ignore_pretrained_weights: [],
  experiment_name: 'default_test',
  seed: seed,
  model_config: {
    base_model: 'ITR_TAPAS',
    ModelClass: 'ITRRagReduceMixModel',
    TokenizerClass: 'DPRQuestionEncoderTokenizer',  // question encoder tokenizer
    TokenizerModelVersion: 'facebook/dpr-question_encoder-single-nq-base',  // question encoder tokenizer version
    DecoderTokenizerClass: 'CustomTapasTokenizer',  // generator tokenizer
    DecoderTokenizerModelVersion: 'google/tapas-base',  // generator tokenizer version

    QueryEncoderModelClass: 'DPRQuestionEncoder',  // question encoder
    QueryEncoderConfigClass: 'DPRConfig',  // question encoder
    // "QueryEncoderModelVersion": "facebook/dpr-question_encoder-single-nq-base",
    QueryEncoderModelVersion: '$DPR_InnerTableRetrieval_wikisql_with_in_batch_neg_sampling_mixed/train/saved_model/step_11604/query_encoder',

    GeneratorModelClass: 'TapasForQuestionAnswering',  // answer generator
    GeneratorConfigClass: 'TapasConfig',
    GeneratorConfigPredefinedSet: 'WTQ',
    GeneratorModelVersion: 'google/tapas-base-finetuned-wikisql-supervised',
    pretrained: 1,
    min_columns: 2,
    loss_ratio: {
      nll_loss: 1,
    },
    modules: [
      'freeze_question_encoder',
    ],
    Ks: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 50, 80, 100],
    num_beams: 5,
    SPECIAL_TOKENS: {
      additional_special_tokens: [],
    },
    DECODER_SPECIAL_TOKENS: {
      additional_special_tokens: [],
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
      module_list: [],
      postprocess_module_list: [],
    },
    output_modules: {
      module_list: [],
      postprocess_module_list: [],
    },
    index_files: index_files,
  },
  data_loader: {
    type: 'DataLoaderForTableQA',
    dummy_dataloader: 0,
    additional: {
      max_source_length: 512,
      max_decoder_source_length: 512,
      max_target_length: 128,
      num_knowledge_passages: 5,
    },
    dataset_modules: {
      module_list: [
        'LoadWikiSQLData',
        'LoadDataLoaders',
      ],
      module_dict: {
        LoadWikiSQLData: {
          type: 'LoadWikiSQLData',
          option: 'default',
          config: {
            preprocess: ['use_original_tapas_data', 'split_table_by_mixed_combination'],
            tapas_path: {
              train: 'TableQA_data/wikisql/tf_examples/train.pkl',
              validation: 'TableQA_data/wikisql/tf_examples/dev.pkl',
              test: 'TableQA_data/wikisql/tf_examples/test.pkl',
            },
            path: {
              train: 'TableQA_data/wikisql/preprocessed_use_original_tapas_data_split_table_by_mixed_combination_train.arrow',
              validation: 'TableQA_data/wikisql/preprocessed_use_original_tapas_data_split_table_by_mixed_combination_validation.arrow',
              test: 'TableQA_data/wikisql/preprocessed_use_original_tapas_data_split_table_by_mixed_combination_test.arrow',
            },
            //"path": {
            //  "train": "TableQA_data/wikisql/preprocessed_split_table_by_mixed_combination_train.arrow",
            //  "validation": "TableQA_data/wikisql/preprocessed_split_table_by_mixed_combination_validation.arrow",
            //  "test": "TableQA_data/wikisql/preprocessed_split_table_by_mixed_combination_test.arrow",
            //}
          },
        },
        LoadDataLoaders: {
          type: 'LoadDataLoaders',
          option: 'default',
          config: {
            train: [
              {
                dataset_type: 'ITRRAGWikiSQLDataset',
                split: 'train',
                use_column: 'wikisql_data',
              },
            ],
            valid: [
              {
                dataset_type: 'ITRRAGWikiSQLDataset',
                split: 'validation',
                use_column: 'wikisql_data',
              },
            ],
            test: [
              {
                dataset_type: 'ITRRAGWikiSQLDataset',
                split: 'test',
                use_column: 'wikisql_data',
              },
            ],
          },
        },
      },
    },
  },
  cuda: 0,
  gpu_device: 0,
  train: {
    type: 'ITRRAGExecutorForTAPAS',
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
      save_top_k_metric: 'valid/ITRRAGWikiSQLDataset.validation/denotation_accuracy',
      weight_decay: 0.01,
      label_smoothing_factor: 0.1,
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
    { name: 'compute_tapas_denotation_accuracy' },
    { name: 'compute_tapas_denotation_accuracy', option: 'valid_samples_only' },
    { name: 'compute_ITR_RAG_retrieval_results' },
  ],
};

std.mergePatch(base_env, override)
