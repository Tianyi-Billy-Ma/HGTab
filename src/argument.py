import sys

sys.dont_write_bytecode = True


import argparse
from pytorch_lightning import Trainer


def parse_args(args_list=None):
    arg_parser = argparse.ArgumentParser(description="A simple argument parser")

    arg_parser.add_argument(
        "config",
        metavar="config_json_file",
        type=str,
        help="Path to the configuration file",
    )
    arg_parser.add_argument(
        "--DATA_FOLDER", type=str, default="", help="The path to data."
    )
    arg_parser.add_argument(
        "--EXPERIMENT_FOLDER",
        type=str,
        default="",
        help="The path to save experiments.",
    )

    arg_parser.add_argument(
        "--mode", type=str, default="", help="create_data/train/test"
    )
    arg_parser.add_argument(
        "--reset",
        action="store_true",
        default=False,
        help="Reset the corresponding folder under the experiment_name",
    )

    arg_parser.add_argument(
        "--experiment_name",
        type=str,
        default="",
        help="Experiment will be saved under /path/to/EXPERIMENT_FOLDER/$experiment_name$.",
    )
    arg_parser.add_argument(
        "--tags", nargs="*", default=[], help="Add tags to the wandb logger"
    )
    arg_parser.add_argument(
        "--modules",
        type=str,
        nargs="+",
        default=[],
        help="Select modules for models. See training scripts for examples.",
    )
    arg_parser.add_argument(
        "--log_prediction_tables",
        action="store_true",
        default=False,
        help="Log prediction tables.",
    )
    arg_parser.add_argument(
        "--override",
        action="store_true",
        default=False,
        help="Danger. Force yes for reset=1",
    )
    arg_parser.add_argument(
        "--disable_wandb_logging",
        action="store_true",
        default=False,
        help="whether to disable wandb logging.",
    )

    # ===== Testing Configuration ===== #
    arg_parser.add_argument("--test_batch_size", type=int, default=-1)
    arg_parser.add_argument("--test_evaluation_name", type=str, default="")

    # arg_parser = Trainer.add_argparse_args(arg_parser)

    arg_parser.add_argument(
        "--opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    # ===== Args for pl===== #
    arg_parser.add_argument("--accelerator", type=str, default="gpu")
    arg_parser.add_argument("--device", type=int, default=1)
    arg_parser.add_argument("--num_sanity_val_steps", type=int, default=2)
    arg_parser.add_argument("--limit_train_batches", default=None)
    arg_parser.add_argument("--limit_val_batches", default=None)
    arg_parser.add_argument("--limit_test_batches", default=None)
    arg_parser.add_argument("--strategy", default="")

    if args_list is None:
        args = arg_parser.parse_args()
    else:
        args = arg_parser.parse_args(args_list)
    return args
