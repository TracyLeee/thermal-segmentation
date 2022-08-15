import argparse
import os
import sys

import torch

from datasets.transforms import *
from trainer import Trainer


def parse_arguments():
    parser = argparse.ArgumentParser(description="Semantic Segmentation with DeepLabV3")
    parser.add_argument(
        "--name",
        type=str,
        help="descriptive name for the training and any saved metadata",
    )
    parser.add_argument(
        "--root",
        type=str,
        default="data",
        help="root directory for the data",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="batch size for training set",
    )
    parser.add_argument(
        "--val-batch-size",
        type=int,
        default=1,
        help="batch size for validation set",
    )
    parser.add_argument(
        "--max-iters",
        type=int,
        default=80000,
        help="max iterations for training",
    )
    parser.add_argument(
        "--info-interval",
        type=int,
        default=10,
        help="interval for logging training information",
    )
    parser.add_argument(
        "--val-interval",
        type=int,
        default=1000,
        help="interval for validation",
    )
    parser.add_argument(
        "--gpu-ids",
        type=int,
        # default=[0, 1],
        default=[0],
        nargs="+",
        help="GPU IDs",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=8,
        help="number of workers for data loader",
    )
    parser.add_argument(
        "--manual-seed",
        type=int,
        default=0,
        help="seed for random number generator",
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        default=20,
        help="number of classes for segmentation",
    )

    if sys.version_info >= (3, 9):
        # argparse.BooleanOptionalAction is a new feature in version 3.9
        parser.add_argument(
            "--drop-last",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="whether to drop the last incomplete batch",
        )
    else:
        parser.add_argument(
            "--drop-last",
            action="store_true",
            help="whether to drop the last incomplete batch",
        )
        parser.add_argument(
            "--no-drop-last",
            dest="drop_last",
            action="store_false",
            help="whether to drop the last incomplete batch",
        )

    parser.add_argument(
        "--optimizer",
        type=str,
        default="sgd",
        choices=["adam", "sgd"],
        help="optimizer for training",
    )
    parser.add_argument(
        "--lr-policy",
        type=str,
        default="step",
        choices=["step"],
        help="learning rate policy for training",
    )
    parser.add_argument(
        "--loss-fn",
        type=str,
        default="cross_entropy",
        choices=["cross_entropy", "focal"],
        help="loss function for training",
    )
    parser.add_argument(
        "--ckpt-directory",
        type=str,
        default="checkpoints",
        help="directory to save checkpoints",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        required=False,
        help="checkpoint (in {ckpt-directory}) to load",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="whether to resume training ({ckpt} must be provided if set true)",
    )

    return parser.parse_args()


def main():
    args = parse_arguments()

    config = dict()

    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, args.gpu_ids))
    config["device"] = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config["name"] = args.name
    config["root"] = args.root
    config["batch_size"] = args.batch_size
    config["val_batch_size"] = args.val_batch_size
    config["max_iters"] = args.max_iters
    config["info_interval"] = args.info_interval
    config["val_interval"] = args.val_interval
    config["num_workers"] = args.num_workers
    config["num_classes"] = args.num_classes
    config["drop_last"] = args.drop_last

    rand_gen = torch.Generator()
    rand_gen.manual_seed(args.manual_seed)

    train_transform = Compose(
        [
            RandomCrop(size=(512, 512), rand_gen=rand_gen),
            RandomHorizontalFlip(prob=0.5, rand_gen=rand_gen),
            ToTensor(),
            Normalize(mean=[0.3135906133471105, 0.3135906133471105, 0.3135906133471105], std=[0.06519744939510239, 0.06519744939510239, 0.06519744939510239]),
        ]
    )

    val_transform = Compose(
        [
            ToTensor(),
            Normalize(mean=[0.3135906133471105, 0.3135906133471105, 0.3135906133471105], std=[0.06519744939510239, 0.06519744939510239, 0.06519744939510239]),
        ]
    )

    config["train_trans"] = train_transform
    config["val_trans"] = val_transform

    config["optim"] = dict()
    config["optim"]["optimizer"] = args.optimizer
    config["optim"]["momentem"] = 0.9
    config["optim"]["weight_dacay"] = 1e-4

    config["lr"] = dict()
    config["lr"]["lr_policy"] = args.lr_policy
    config["lr"]["base_lr"] = 0.01
    config["lr"]["step_size"] = 10000
    config["lr"]["gamma"] = 0.1

    config["loss_fn"] = args.loss_fn

    config["immutable"] = dict()
    config["immutable"]["ckpt_directory"] = args.ckpt_directory
    config["immutable"]["ckpt"] = args.ckpt
    config["immutable"]["resume"] = args.resume

    model = Trainer(config)
    model.train()


if __name__ == "__main__":
    main()
