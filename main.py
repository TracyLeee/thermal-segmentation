import argparse
import os
import sys

import torch

from dataset_utils.transforms import *
from tester import Tester
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
    parser.add_argument(
        "--test",
        action="store_true",
        help="whether to test the model ({ckpt} must be provided if set true)",
    )
    parser.add_argument(
        "--seg-results-directory",
        type=str,
        default="seg_results",
        help="directory to save segmentation results",
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
    config["drop_last"] = args.drop_last

    rand_gen = torch.Generator()
    rand_gen.manual_seed(args.manual_seed)

    train_transform = Compose(
        [
            RandomCrop(size=(512, 512), rand_gen=rand_gen),
            RandomHorizontalFlip(prob=0.5, rand_gen=rand_gen),
            ToTensor(),
            Normalize(
                mean=[0.3135906133471105, 0.3135906133471105, 0.3135906133471105],
                std=[0.06519744939510239, 0.06519744939510239, 0.06519744939510239],
            ),
        ]
    )

    val_transform = Compose(
        [
            ToTensor(),
            Normalize(
                mean=[0.3304213428873094, 0.3304213428873094, 0.3304213428873094],
                std=[0.057050643620181946, 0.057050643620181946, 0.057050643620181946],
            ),
        ]
    )

    test_transform = Compose(
        [
            ToTensor(),
            Normalize(
                mean=[0.3304213428873094, 0.3304213428873094, 0.3304213428873094],
                std=[0.057050643620181946, 0.057050643620181946, 0.057050643620181946],
            ),
        ]
    )

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

    config["class_def"] = {
        0: {"name": "undefined", "color": (0, 0, 0)},
        7: {"name": "road", "color": (128, 64, 128)},
        8: {"name": "sidewalk", "color": (244, 35, 232)},
        11: {"name": "building", "color": (70, 70, 70)},
        12: {"name": "wall", "color": (102, 102, 156)},
        13: {"name": "fense", "color": (190, 153, 153)},
        17: {"name": "pole", "color": (153, 153, 153)},
        19: {"name": "traffic light", "color": (250, 170, 30)},
        20: {"name": "traffic sign", "color": (220, 220, 0)},
        21: {"name": "vegetation", "color": (107, 142, 35)},
        22: {"name": "terrain", "color": (152, 251, 152)},
        23: {"name": "sky", "color": (70, 130, 180)},
        24: {"name": "person", "color": (220, 20, 60)},
        25: {"name": "rider", "color": (255, 0, 0)},
        26: {"name": "car", "color": (0, 0, 142)},
        27: {"name": "truck", "color": (0, 0, 70)},
        28: {"name": "bus", "color": (0, 60, 100)},
        31: {"name": "train", "color": (0, 80, 100)},
        32: {"name": "motorcycle", "color": (0, 0, 230)},
        33: {"name": "bicycle", "color": (119, 11, 32)},
    }
    config["class_list"] = list(config["class_def"].keys())
    config["num_classes"] = len(config["class_def"])

    config["class_blue_map"] = dict()
    config["class_green_map"] = dict()
    config["class_red_map"] = dict()
    for class_id, class_info in config["class_def"].items():
        config["class_blue_map"][class_id] = class_info["color"][2]
        config["class_green_map"][class_id] = class_info["color"][1]
        config["class_red_map"][class_id] = class_info["color"][0]

    config["color_palette"] = {
        "undefined/curb": (0, 0, 0),
        "road/parking": (128 / 255, 64 / 255, 128 / 255),
        "sidewalk": (244 / 255, 35 / 255, 232 / 255),
        "building": (70 / 255, 70 / 255, 70 / 255),
        "fense": (190 / 255, 153 / 255, 153 / 255),
        "pole/signs": (153 / 255, 153 / 255, 153 / 255),
        "vegetation": (107 / 255, 142 / 255, 35 / 255),
        "terrain": (152 / 255, 251 / 255, 152 / 255),
        "sky": (70 / 255, 130 / 255, 180 / 255),
        "person/rider": (220 / 255, 20 / 255, 60 / 255),
        "car/truck/bus/train": (0, 0, 142 / 255),
        "motorcycle/bicycle": (0, 0, 230 / 255),
        "wall/background": (102 / 255, 102 / 255, 156 / 255),
    }

    config["immutable"] = dict()
    config["immutable"]["ckpt_directory"] = args.ckpt_directory
    config["immutable"]["ckpt"] = args.ckpt
    config["immutable"]["resume"] = args.resume
    config["immutable"]["test"] = args.test
    config["immutable"]["seg_results_directory"] = args.seg_results_directory

    if args.test:
        config["test_trans"] = test_transform
        model = Tester(config)
        model.test()
    else:
        config["train_trans"] = train_transform
        config["val_trans"] = val_transform
        model = Trainer(config)
        model.train()


if __name__ == "__main__":
    main()
