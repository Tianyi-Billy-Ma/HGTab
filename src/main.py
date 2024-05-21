import sys

sys.dont_write_bytecode = True

import os
import wandb
import glob
import json

from pprint import pprint
from easydict import EasyDict

from argument import parse_args

import logging
from logging.handlers import RotatingFileHandler
from logging import Formatter

logger = logging.getLogger(__name__)


import torch


from pytorch_lightning import seed_everything, Trainer
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import torch.distributed as dist

from utils.config_system import process_config
from utils.dirs import reset_folders, create_dirs, reset_wandb_runs
from utils.seed import set_seed
from utils.metrics_log_callback import MetricsHistoryLogger


from trainers import *
from data_loader_manager import *


def get_checkpoint_model_path(
    saved_model_path, load_epoch=-1, load_best_model=False, load_model_path=""
):
    if load_model_path:
        path_save_model = load_model_path
        if not os.path.exists(path_save_model):
            raise FileNotFoundError("Model file not found: {}".format(path_save_model))
    else:
        if load_best_model:
            file_name = "best.ckpt"
        else:
            if load_epoch == -1:
                file_name = "last.ckpt"
            else:
                file_name = "model_step_{}.ckpt".format(load_epoch)

        path_save_model = os.path.join(saved_model_path, file_name)

        file_names = glob.glob(f"{saved_model_path}/*.ckpt", recursive=True)
        logger.info(f"available checkpoints: {file_names}")

        if not os.path.exists(path_save_model):
            logger.warning(
                "No checkpoint exists from '{}'. Skipping...".format(path_save_model)
            )
            logger.info("**First time to train**")
            return ""  # return empty string to indicate that no model is loaded
        else:
            logger.info("Loading checkpoint from '{}'".format(path_save_model))
    return path_save_model


def initialization(args):
    # Check if the mode is valid
    assert args.mode in ["create_data", "train", "test", "run"]

    # ======================= Process Config =======================
    config = process_config(args)

    print("==" * 30 + "\n\n" + "CONFIGURATION:\n\n" + f"{config}\n\n")
    if config is None:
        return None
    dirs = [config.log_path]

    dirs += (
        [config.saved_model_path, config.imgs_path, config.tensorboard_path]
        if config.mode == "train"
        else [config.imgs_path, config.results_path]
    )

    delete_confirm = "n"
    if config.reset and config.mode == "train":
        # Reset all the folders
        print("You are deleting following dirs: ", dirs, "input y to continue")
        if config.args.override:
            delete_confirm = "y"
        else:
            delete_confirm = input()
        if delete_confirm == "y":
            reset_folders(dirs)
            # Reset load epoch after reset
            config.train.load_epoch = 0
        else:
            print("reset cancelled.")

    create_dirs(dirs)
    print("==" * 30 + "\n\n" + "CREATED DIRS:\n\n" + f"{dirs}\n\n")

    # ======================= Setup Logger =======================
    log_file_format = "[%(levelname)s] - %(asctime)s - %(name)s : %(message)s (in %(pathname)s:%(lineno)d)"
    log_console_format = "[%(levelname)s] - %(name)s : %(message)s"

    # Main logger
    main_logger = logging.getLogger()
    main_logger.setLevel(logging.DEBUG)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(Formatter(log_console_format))
    from utils.color_logging import CustomFormatter

    custom_output_formatter = CustomFormatter(custom_format=log_console_format)
    console_handler.setFormatter(custom_output_formatter)

    info_file_handler = RotatingFileHandler(
        os.path.join(config.log_path, "info.log"), maxBytes=10**6, backupCount=5
    )
    info_file_handler.setLevel(logging.INFO)
    info_file_handler.setFormatter(Formatter(log_file_format))

    exp_file_handler = RotatingFileHandler(
        os.path.join(config.log_path, "debug.log"), maxBytes=10**6, backupCount=5
    )
    exp_file_handler.setLevel(logging.DEBUG)
    exp_file_handler.setFormatter(Formatter(log_file_format))

    exp_errors_file_handler = RotatingFileHandler(
        os.path.join(config.log_path, "error.log"), maxBytes=10**6, backupCount=5
    )
    exp_errors_file_handler.setLevel(logging.WARNING)
    exp_errors_file_handler.setFormatter(Formatter(log_file_format))

    main_logger.addHandler(console_handler)
    main_logger.addHandler(info_file_handler)
    main_logger.addHandler(exp_file_handler)
    main_logger.addHandler(exp_errors_file_handler)

    # setup a hook to log unhandled exceptions
    def handle_exception(exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            if wandb.run is not None:
                logger.error(f"Attempting to stop the wandb run {wandb.run}")
                wandb.finish()  # stop wandb if keyboard interrupt is raised
            sys.__excepthook__(exc_type, exc_value, exc_traceback)

        logger.error(
            f"Uncaught exception: {exc_type} --> {exc_value}",
            exc_info=(exc_type, exc_value, exc_traceback),
        )
        if not config.args.disable_wandb_logging and wandb.run is not None:
            wandb.finish()
            # subprocess.run(["wandb", "sync", "--sync-all"])
            logger.info("Force sync wandb files")

    sys.excepthook = handle_exception

    if not config.args.disable_wandb_logging:
        # setup wandb
        WANDB_CACHE_DIR = config.WANDB.pop("CACHE_DIR")
        if WANDB_CACHE_DIR:
            os.environ["WANDB_CACHE_DIR"] = WANDB_CACHE_DIR
        else:
            os.environ["WANDB_CACHE_DIR"] = ""

        WANDB_DIR = config.WANDB.pop("DIR")
        if WANDB_DIR:
            os.environ["WANDB_DIR"] = WANDB_DIR
        else:
            os.environ["WANDB_DIR"] = ""

        config.WANDB.dir = os.environ["WANDB_DIR"]

        # add base_model as a tag
        config.WANDB.tags.append(config.model_config.base_model)
        # add modules as tags
        config.WANDB.tags.extend(config.model_config.modules)

        all_runs = wandb.Api(timeout=19).runs(
            path=f"{config.WANDB.entity}/{config.WANDB.project}",
            filters={"config.experiment_name": config.experiment_name},
        )
        if config.reset and config.mode == "train" and delete_confirm == "y":
            reset_wandb_runs(all_runs)
            config.WANDB.name = config.experiment_name
        else:
            if len(all_runs) > 0:
                config.WANDB.id = all_runs[0].id
                config.WANDB.resume = "must"
                config.WANDB.name = config.experiment_name
            else:
                config.WANDB.name = config.experiment_name
    logger.info(f"Initialization done with the config: {str(config)}")
    return config


def main(arg_list=None):
    args = parse_args(arg_list)
    print("==" * 30 + "\n\n" + "ARGUMENTS:\n\n" + f"{args}\n\n")
    config = initialization(args)
    if config is None:
        raise ValueError("No config file is obtained, exiting...")

    args = config.args

    pprint(config)

    if config.seed:
        set_seed(config.seed)
        seed_everything(config.seed)
        logger.info(f"All seeds have been set to {config.seed}")

    DataLoaderWrapper = globals()[config.data_loader.type]

    assert (
        DataLoaderWrapper is not None
    ), f"Data Loader {config.data_loader.type} not found"

    data_loader_manager = DataLoaderWrapper(config)
    if config.mode == "create_data":
        data_loader_manager.build_dataset()
        logger.info(f"Finished building data, exiting main program...")
        return

    # Default logger
    tb_logger = TensorBoardLogger(
        save_dir=config.tensorboard_path, name=config.experiment_name
    )

    callback_list = []
    # Checkpoint Callback
    checkpoint_callback = ModelCheckpoint(
        dirpath=config.saved_model_path,
        # every_n_train_steps=config.train.save_interval,
        save_top_k=config.train.additional.save_top_k,
        monitor=config.train.additional.save_top_k_metric
        if "save_top_k_metric" in config.train.additional.keys()
        else None,
        mode=config.train.additional.save_top_k_mode,
        filename="model_step_{step}",
        save_last=True,
        verbose=True,
        auto_insert_metric_name=False,
        save_on_train_epoch_end=False,
    )
    callback_list.append(checkpoint_callback)

    # Early Stopping Callback
    if (
        "save_top_k_metric" in config.train.additional.keys()
        and config.train.additional.get("early_stop_patience", 0) > 0
    ):
        early_stop_callback = EarlyStopping(
            monitor=config.train.additional.save_top_k_metric,
            patience=config.train.additional.early_stop_patience,
            verbose=True,
            mode=config.train.additional.save_top_k_mode,
        )
        callback_list.append(early_stop_callback)

    metrics_history_logger = MetricsHistoryLogger()

    # Get plugins
    plugin_names = config.train.additional.plugins
    plugins = [globals()[plugin_name]() for plugin_name in plugin_names]

    all_loggers = [tb_logger, metrics_history_logger]
    if config.args.disable_wandb_logging:
        # Disable logging wandb tables
        config.args.log_prediction_tables = False
    else:
        # Wandb logger
        logger.info(
            "init wandb logger with the following settings: {}".format(config.WANDB)
        )
        wandb_logger = WandbLogger(config=config, **config.WANDB)
        all_loggers.append(wandb_logger)

    additional_args = {
        "accumulate_grad_batches": config.train.additional.gradient_accumulation_steps,
        "default_root_dir": config.saved_model_path,
        "max_epochs": config.train.epochs,
        "limit_train_batches": 2
        if args["limit_train_batches"] is None and config.data_loader.dummy_dataloader
        else args["limit_train_batches"],
        "limit_val_batches": 2
        if args["limit_val_batches"] is None and config.data_loader.dummy_dataloader
        else args["limit_val_batches"],
        "limit_test_batches": 2
        if args["limit_test_batches"] is None and config.data_loader.dummy_dataloader
        else args["limit_test_batches"],
        "logger": all_loggers,
        "callbacks": callback_list,
        "plugins": plugins,
        "log_every_n_steps": 10,
        "check_val_every_n_epoch": None,
        "val_check_interval": config.valid.step_size
        * config.train.additional.gradient_accumulation_steps,  # this is to use global_step as the interval number: global_step * grad_accumulation = batch_idx (val_check_interval is based on batch_idx)
        # 'accelerator': "cpu",
        # 'strategy': "ddp",
        # 'devices': 2,
    }

    if args.strategy == "ddp":
        from pytorch_lightning.strategies import DDPStrategy

        additional_args["strategy"] = DDPStrategy(find_unused_parameters=True)

    trainer_args = args.copy()
    trainer_args.update(additional_args)
    trainer_args = EasyDict(
        {
            k: v
            for k, v in trainer_args.items()
            if k
            in [
                "accelerator",
                "strategy",
                "devices",
                "num_nodes",
                "precision",
                "logger",
                "callbacks",
                "fast_dev_run",
                "max_epochs",
                "min_epochs",
                "max_steps",
                "min_steps",
                "max_time",
                "limit_train_batches",
                "limit_val_batches",
                "limit_test_batches",
                "limit_predict_batches",
                "overfit_batches",
                "val_check_interval",
                "check_val_every_n_epoch",
                "num_sanity_val_steps",
                "log_every_n_steps",
                "enable_checkpointing",
                "enable_progress_bar",
                "enable_model_summary",
                "accumulate_grad_batches",
                "gradient_clip_val",
                "gradient_clip_algorithm",
                "deterministic",
                "benchmark",
                "inference_mode",
                "use_distributed_sampler",
                "profiler",
                "detect_anomaly",
                "barebones",
                "plugins",
                "sync_batchnorm",
                "reload_dataloaders_every_n_epochs",
                "default_root_dir",
            ]
        }
    )

    trainer = Trainer(**vars(trainer_args))
    logger.info(f"arguments passed to trainer: {str(args)}")
    logger.info(f"additional arguments passed to trainer: {str(additional_args)}")

    # Find checkpoints in saved_model_path
    if config.mode == "train":
        checkpoint_to_load = get_checkpoint_model_path(
            saved_model_path=config.saved_model_path,
            load_model_path=config.train.load_model_path,
            load_epoch=config.train.load_epoch,
            load_best_model=config.train.load_best_model,
        )
        if not checkpoint_to_load:
            logger.warning("No checkpoint found. Starting from scratch.")
            checkpoint_to_load = None
    else:
        checkpoint_to_load = get_checkpoint_model_path(
            saved_model_path=config.saved_model_path,
            load_model_path=config.test.load_model_path,
            load_epoch=config.test.load_epoch,
            load_best_model=config.test.load_best_model,
        )
        if not checkpoint_to_load:
            logger.warning("No checkpoint found. Please check your config file.")

    # init data loader manager
    data_loader_manager.build_dataset()

    torch.cuda.empty_cache()

    if config.mode == "train":
        # init train excecutor
        Train_Executor = globals()[config.train.type]
        executor = Train_Executor(config, data_loader_manager)
        # After Initialization, save config files
        with open(
            os.path.join(config.experiment_path, "config.jsonnet"), "w"
        ) as config_file:
            save_config = config.copy()
            # save_config.pop('device') # Not serialisable
            json.dump(save_config, config_file, indent=4)
            logger.info(
                f"config file was successfully saved to {config.experiment_path} for future use."
            )
        # Start training
        trainer.fit(
            executor,
            ckpt_path=checkpoint_to_load,
        )

    else:
        # init train excecutor
        Train_Executor = globals()[config.train.type]
        executor = Train_Executor(config, data_loader_manager)
        # Start testing
        trainer.test(
            executor,
            ckpt_path=checkpoint_to_load if checkpoint_to_load else None,
        )

    if not config.args.disable_wandb_logging:
        logger.info("task finished. finishing wandb process...")
        wandb.finish()


if __name__ == "__main__":
    llama = [
        "configs/wikiTQ/Llama3_base.jsonnet",
        "--accelerator",
        "gpu",
        "--device",
        "1",
        "--strategy",
        "ddp",
        "--experiment_name",
        # "DPR_InnerTableRetrieval_wikisql_with_in_batch_neg_sampling_mixed",
        "Llama_3_8B_wikitq",
        "--mode",
        "test",
        "--override",
        "--disable_wandb_logging",
        "--opts",
        "train.batch_size=1",
        # "train.scheduler=None",
        # "train.epochs=5",
        # "train.lr=0.00001",
        # "train.additional.gradient_accumulation_steps=4",
        # "train.additional.warmup_steps=200",
        # "train.additional.early_stop_patience=8",
        # "train.additional.save_top_k=3",
        # "valid.batch_size=8",
        # "test.batch_size=8",
        # "valid.step_size=200",
        "reset=1",
    ]

    dpr_ITR_mix_wtq = [
        "configs/wikiTQ/dpr_ITR_mix_wtq.jsonnet",
        "--accelerator",
        "gpu",
        "--device",
        "1",
        "--strategy",
        "ddp",
        "--experiment_name",
        "DPR_InnerTableRetrieval_wikisql_with_in_batch_neg_sampling_mixed",
        # "Llama_3_8B_wikitq",
        "--mode",
        "test",
        # "--override",
        "--disable_wandb_logging",
        "--opts",
        # "train.batch_size=1",
        # "train.scheduler=None",
        # "train.epochs=5",
        # "train.lr=0.00001",
        # "train.additional.gradient_accumulation_steps=4",
        # "train.additional.warmup_steps=200",
        # "train.additional.early_stop_patience=8",
        # "train.additional.save_top_k=3",
        # "valid.batch_size=8",
        # "test.batch_size=8",
        # "valid.step_size=200",
        "reset=0",
    ]

    # main(arg_list2)

    tapex_wikitq = [
        "configs/wikiTQ/tapex_base.jsonnet",
        "--accelerator",
        "gpu",
        "--device",
        "1",
        "--strategy",
        "ddp",
        "--experiment_name",
        "finetune_tapex_large_on_WikiTQ_smoothing_0.1",
        "--mode",
        "train",
        "--opts",
        "train.batch_size=1",
        "train.scheduler=linear",
        "train.epochs=20",
        "train.lr=0.00003",
        "train.additional.gradient_accumulation_steps=4",
        "train.additional.warmup_steps=1000",
        "train.additional.early_stop_patience=6",
        "train.additional.save_top_k=3",
        "train.save_interval=1000",
        "valid.batch_size=4",
        "test.batch_size=4",
        "data_loader.dummy_dataloader=0",
        "train.additional.label_smoothing_factor=0.1",
    ]

    main(tapex_wikitq)
