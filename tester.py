import os

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils import data
from torchvision.models.segmentation import deeplabv3_resnet50
from tqdm import tqdm

from dataset_utils.thermal_loader import ThermalDataset
from metrics.evaluation import SegmentMetrics


class Tester(object):
    def __init__(self, config: dict):
        self.config = config

        self.ckpt_directory = self.config["immutable"]["ckpt_directory"]
        self.ckpt = self.config["immutable"]["ckpt"]
        self.resume = self.config["immutable"]["resume"]

        self.net = None
        self.loss_fn = None

        self._init_model()
        print(self.config)

        self.name = self.config["name"]

        self.class_list = self.config["class_list"]
        self.num_classes = self.config["num_classes"]
        self.batch_size = self.config["batch_size"]

        self.test_set = ThermalDataset(
            self.config["root"], "test", self.config["test_trans"]
        )
        self.test_loader = data.DataLoader(
            self.test_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.config["num_workers"],
        )

        # torch._C.Generator object cannot be deepcopied...
        # This could be a feature expected to be added later in pytorch
        # https://github.com/pytorch/pytorch/issues/43672
        self.config.pop("test_trans", None)

        self.device = self.config["device"]

        self.metrics = SegmentMetrics(np.arange(self.num_classes))

        if not os.path.exists(self.config["immutable"]["seg_results_directory"]):
            os.mkdir(self.config["immutable"]["seg_results_directory"])
        
        self.seg_results_directory = os.path.join(self.config["immutable"]["seg_results_directory"], self.name)
        if not os.path.exists(self.seg_results_directory):
            os.mkdir(self.seg_results_directory)

    def _init_model(self):
        state_dict = None

        if self.ckpt is not None:
            ckpt_path = os.path.join(self.ckpt_directory, self.ckpt)

            if not os.path.isfile(ckpt_path) or not self.ckpt.endswith(".pth"):
                raise TypeError("Invalid checkpoint type!")

            state_dict = torch.load(ckpt_path, map_location=torch.device("cpu"))

            self.net = deeplabv3_resnet50(
                num_classes=self.config["num_classes"], pretained=False
            )
            self.net = torch.nn.DataParallel(self.net)
            self.net.load_state_dict(state_dict["net_state"])
        else:
            raise ValueError("Checkpoint must be provided!")

        self.net.to(self.config["device"])

        if self.config["loss_fn"] == "cross_entropy":
            self.loss_fn = torch.nn.CrossEntropyLoss(reduction="mean")

        if state_dict is not None:
            del state_dict

    def _decode_label(self, label):
        label_map = dict(zip(np.arange(self.num_classes), self.class_list))

        return np.vectorize(label_map.get)(label)
    
    def _visualize_label(self, label):
        colored_label = np.empty((*label.shape, 3))
        colored_label[..., 0] = np.vectorize(self.config["class_blue_map"].get)(label)
        colored_label[..., 1] = np.vectorize(self.config["class_green_map"].get)(label)
        colored_label[..., 2] = np.vectorize(self.config["class_red_map"].get)(label)

        return colored_label
    
    def _visualize_results(self, labels, idx):
        batch_size = labels.shape[0]

        for i in range(batch_size):
            label = labels[i, ...]
            label = self._decode_label(label)
            colored_label = self._visualize_label(label)
            label_filename = os.path.basename(self.test_set.label_list[idx + i])
            
            _, ax = plt.subplots()
            plt.imshow(colored_label)
            patch_gen = lambda label, color: mpatches.Patch(label=label, color=color)
            ax.legend(handles=[patch_gen(label, color) for label, color in self.config["color_palette"].items()])
            ax.set_axis_off()
            plt.savefig(os.path.join(self.seg_results_directory, label_filename))
        
        return idx + batch_size


    def _test(self):
        self.net.eval()
        self.metrics.reset()

        with torch.no_grad():
            idx = 0

            for (images, labels) in tqdm(self.test_loader):
                images = images.to(self.device, dtype=torch.float)
                labels = labels.to(self.device, dtype=torch.long)

                outputs = self.net(images)
                outputs = outputs["out"]
                loss = self.loss_fn(outputs, labels)

                predictions = outputs.detach().max(dim=1)[1].cpu().numpy()
                labels = labels.cpu().numpy()

                self.metrics.update(labels, predictions, loss, self.batch_size)

                idx = self._visualize_results(labels, idx)

            overall_iou = self.metrics.overall_iou
            overall_acc = self.metrics.overall_acc
            avg_loss = self.metrics.avg_loss

            print(f"IoU: {overall_iou}, ACC: {overall_acc}, loss: {avg_loss}")
    
    def test(self):
        self._test()
