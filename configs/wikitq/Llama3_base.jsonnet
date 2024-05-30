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

local seed = 3;


local override = {
  platform_type: 'pytorch',

  experiment_name: 'Llama_3_8B_wikitq',
  seed: seed,
  model_config: {
    base_model: 'Meta-Llama-3-8B',
    ModelClass: 'AutoModel',
    TokenizerClass: 'AutoTokenizer',
    TokenizerModelVersion: 'meta-llama/Meta-Llama-3-8B',
    ConfigClass: 'AutoConfig',
    ModelVersion: 'meta-llama/Meta-Llama-3-8B',
    ConfigModelVersion: 'meta-llama/Meta-Llama-3-8B',
    modules: [],
    SPECIAL_TOKENS: {
      pad_token: '[PAD]',
      additional_special_tokens: [],
    },
    input_modules: {
      module_list: [
        {
          type: 'QuestionInput',
          option: 'default',
          separation_tokens: { start: '', end: '' },
        },
        {
          type: 'TableInput',
          option: 'default',
        },
      ],
      postprocess_module_list: [
        { type: 'PostProcessInputTokenization', option: 'default' },
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
          config: {
            preprocess: ['transform_to_sqa_format'],
          },
        },
        LoadDataLoaders: {
          type: 'LoadDataLoaders',
          option: 'default',
          config: {
            train: [
              {
                dataset_type: 'WikiTQDataset',
                split: 'train',
                use_column: 'wtq_data',
              },
            ],
            valid: [
              {
                dataset_type: 'WikiTQDataset',
                split: 'validation',
                use_column: 'wtq_data',
              },
              {
                dataset_type: 'WikiTQDataset',
                split: 'test',
                use_column: 'wtq_data',
              },
            ],
            test: [
              {
                dataset_type: 'WikiTQDataset',
                split: 'validation',
                use_column: 'wtq_data',
              },
              {
                dataset_type: 'WikiTQDataset',
                split: 'test',
                use_column: 'wtq_data',
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
    type: 'Llama3Executor',
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
      save_top_k_metric: 'valid/WikiTQDataset.validation/denotation_accuracy',
      weight_decay: 0,
      label_smoothing_factor: 0,
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
    { name: 'compute_llama3_denotation_accuracy' },
    { name: 'compute_llama3_denotation_accuracy', option: 'valid_samples_only' },
  ],
};


std.mergePatch(base_env, override)
