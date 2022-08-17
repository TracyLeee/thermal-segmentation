import copy
import os

import numpy as np
import torch
from torch.utils import data
from torchvision.models.segmentation import deeplabv3_resnet50
from tqdm import tqdm

from dataset_utils.thermal_loader import ThermalDataset
from metrics.evaluation import SegmentMetricLoss, SegmentMetrics


class Trainer(object):
    def __init__(self, config: dict):
        self.config = config

        self.ckpt_directory = self.config["immutable"]["ckpt_directory"]
        self.ckpt = self.config["immutable"]["ckpt"]
        self.resume = self.config["immutable"]["resume"]

        self.net = None
        self.optimizer = None
        self.scheduler = None
        self.loss_fn = None

        self._init_model()
        print(self.config)

        self.name = self.config["name"]

        self.class_list = self.config["class_list"]
        self.num_classes = self.config["num_classes"]
        self.batch_size = self.config["batch_size"]
        self.val_batch_size = self.config["val_batch_size"]

        self.train_set = ThermalDataset(
            self.config["root"],
            "train",
            self.config["train_trans"],
        )
        self.train_loader = data.DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.config["num_workers"],
            drop_last=self.config["drop_last"],
        )
        self.val_set = ThermalDataset(
            self.config["root"], "val", self.config["val_trans"]
        )
        self.val_loader = data.DataLoader(
            self.val_set,
            batch_size=self.val_batch_size,
            shuffle=True,
            num_workers=self.config["num_workers"],
        )

        # torch._C.Generator object cannot be deepcopied...
        # This could be a feature expected to be added later in pytorch
        # https://github.com/pytorch/pytorch/issues/43672
        self.config.pop("train_trans", None)
        self.config.pop("val_trans", None)

        self.device = self.config["device"]

        self.max_iters = self.config["max_iters"]
        self.info_interval = self.config["info_interval"]
        self.val_interval = self.config["val_interval"]
        if not self.resume:
            self.epochs = 0
            self.iters = 0

        self.metrics = SegmentMetrics(np.arange(self.num_classes))
        self.metric_loss = SegmentMetricLoss()
        if not self.resume:
            self.max_overall_iou = 0.0
            self.min_avg_loss = float("inf")

    def _merge_config(self, config: dict):
        """Merges config dict from the checkpoint into current config dict.
        Parameters under "immutable" key are unchanged.

        """
        # torch._C.Generator object cannot be deepcopied...
        # This could be a feature expected to be added later in pytorch
        # https://github.com/pytorch/pytorch/issues/43672
        train_transform = self.config.pop("train_trans", None)
        val_transform = self.config.pop("val_trans", None)

        new_config = copy.deepcopy(self.config)
        new_config.update(config)

        new_config["train_trans"] = train_transform
        new_config["val_trans"] = val_transform

        return new_config

    def _init_model(self):
        state_dict = None

        if self.ckpt is not None:
            ckpt_path = os.path.join(self.ckpt_directory, self.ckpt)

            if not os.path.isfile(ckpt_path) or not self.ckpt.endswith(".pth"):
                raise TypeError("Invalid checkpoint type!")

            state_dict = torch.load(ckpt_path, map_location=torch.device("cpu"))

            self.config = self._merge_config(state_dict["config"])
            self.net = deeplabv3_resnet50(
                num_classes=self.config["num_classes"], pretained=False
            )
            self.net = torch.nn.DataParallel(self.net)
            self.net.load_state_dict(state_dict["net_state"])
        else:
            self.net = deeplabv3_resnet50(
                num_classes=self.config["num_classes"], pretained=False
            )
            self.net = torch.nn.DataParallel(self.net)

        self.net.to(self.config["device"])

        if self.config["optim"]["optimizer"] == "sgd":
            self.optimizer = torch.optim.SGD(
                self.net.parameters(),
                lr=self.config["lr"]["base_lr"],
                momentum=self.config["optim"]["momentem"],
                weight_decay=self.config["optim"]["weight_dacay"],
            )
        else:
            raise RuntimeError("optimizer not implemented!")

        if self.config["lr"]["lr_policy"] == "step":
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.config["lr"]["step_size"],
                gamma=self.config["lr"]["gamma"],
            )
        else:
            raise RuntimeError("lr policy not implemented!")

        if self.config["loss_fn"] == "cross_entropy":
            self.loss_fn = torch.nn.CrossEntropyLoss(reduction="mean")

        if self.resume:
            if self.ckpt is None:
                raise ValueError(
                    "Attemping to resume the training but checkpoint not given!"
                )

            self.optimizer.load_state_dict(state_dict["optimizer_state"])
            self.scheduler.load_state_dict(state_dict["scheduler_state"])
            self.epochs = state_dict["epochs"]
            self.iters = state_dict["iters"]
            self.max_overall_iou = state_dict["max_overall_iou"]
            self.min_avg_loss = state_dict["min_avg_loss"]

        if state_dict is not None:
            del state_dict

    def _save_checkpoint(self, suffix: str = "latest"):
        if not os.path.exists(self.ckpt_directory):
            os.mkdir(self.ckpt_directory)

        state_dict = dict()

        state_dict["epochs"] = self.epochs
        state_dict["iters"] = self.iters
        state_dict["net_state"] = self.net.state_dict()
        state_dict["optimizer_state"] = self.optimizer.state_dict()
        state_dict["scheduler_state"] = self.scheduler.state_dict()
        state_dict["max_overall_iou"] = self.max_overall_iou
        state_dict["min_avg_loss"] = self.min_avg_loss
        saved_config = copy.deepcopy(self.config)
        saved_config.pop("immutable", None)
        state_dict["config"] = saved_config

        torch.save(
            state_dict, os.path.join(self.ckpt_directory, f"{self.name}_{suffix}.pth")
        )

    def _train(self):
        print("Training...")

        while True:
            self.net.train()

            for (images, labels) in self.train_loader:
                images = images.to(self.device, dtype=torch.float)
                labels = labels.to(self.device, dtype=torch.long)

                self.optimizer.zero_grad()
                outputs = self.net(images)
                outputs = outputs["out"]
                loss = self.loss_fn(outputs, labels)
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()

                self.metric_loss.update(loss.detach().cpu().numpy(), self.batch_size)

                if self.iters % self.info_interval == self.info_interval - 1:
                    print(
                        f"[epoch: {self.epochs}, iters: {self.iters}/{self.max_iters - 1}] average loss: {self.metric_loss.avg_loss}"
                    )
                    self.metric_loss.reset()

                if self.iters % self.val_interval == self.val_interval - 1:
                    print("\nValidating...")
                    self._val()
                    print("Validation done.\n")

                self.iters += 1

                if self.iters >= self.max_iters:
                    return

            self.epochs += 1

    def _val(self):
        self.net.eval()
        self.metrics.reset()

        with torch.no_grad():
            for (images, labels) in tqdm(self.val_loader):
                images = images.to(self.device, dtype=torch.float)
                labels = labels.to(self.device, dtype=torch.long)

                outputs = self.net(images)
                outputs = outputs["out"]
                loss = self.loss_fn(outputs, labels)

                predictions = outputs.detach().max(dim=1)[1].cpu().numpy()
                labels = labels.cpu().numpy()

                # (1, 1400, 650) --> (1400, 650)?
                self.metrics.update(labels, predictions, loss, self.val_batch_size)

            overall_iou = self.metrics.overall_iou
            overall_acc = self.metrics.overall_acc
            avg_loss = self.metrics.avg_loss

            print(f"IoU: {overall_iou}, ACC: {overall_acc}, loss: {avg_loss}")

            self._save_checkpoint("latest")

            if overall_iou > self.max_overall_iou:
                print(f"IoU: {self.max_overall_iou} -> {overall_iou}")
                self.max_overall_iou = overall_iou
                self._save_checkpoint("max_iou")

            if avg_loss < self.min_avg_loss:
                print(f"Loss: {self.min_avg_loss} -> {avg_loss}")
                self.min_avg_loss = avg_loss
                self._save_checkpoint("min_loss")

        self.net.train()

    def train(self):
        self._train()
