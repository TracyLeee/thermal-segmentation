import os

import cv2
import numpy as np
from PIL import Image
from torch.utils import data


class ThermalDataset(data.Dataset):
    """Dataset for thermal images

    :param data: _description_
    """

    def __init__(self, root, split="train", transform=None):
        self.root = root

        if split not in ["train", "test", "val"]:
            raise ValueError("Invalid split! ('train', 'test' or 'val')")

        self.split = split
        self.image_dir = os.path.join(*[root, split, "images"])
        self.label_dir = os.path.join(*[root, split, "labels"])

        if not os.path.isdir(self.image_dir) or not os.path.isdir(self.label_dir):
            raise RuntimeError("Invalid dataset structure!")

        self.image_list, self.label_list = self._generate_data_list()

        self.transform = transform

        self.class_list = [
            0,
            7,
            8,
            11,
            12,
            13,
            17,
            19,
            20,
            21,
            22,
            23,
            24,
            25,
            26,
            27,
            28,
            31,
            32,
            33,
        ]
        self.num_classes = len(self.class_list)

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        image = Image.open(self.image_list[index]).convert("RGB")
        label = cv2.imread(self.label_list[index], -1)
        label = self._encode_label(label)

        if self.transform:
            image, label = self.transform((image, label))

        return image, label

    def _generate_data_list(self):
        image_list = list()
        label_list = list()

        for image in os.listdir(self.image_dir):
            image_list.append(os.path.join(self.image_dir, image))
            data_id = image.split("fl_ir_aligned_")[1]
            label_list.append(
                os.path.join(self.label_dir, f"fl_ir_aligned_labels_{data_id}")
            )
            # label_list.append(os.path.join(self.label_dir, f"fl_ir_aligned_{data_id}"))

        return image_list, label_list

    def _encode_label(self, label):
        label_map = dict(zip(self.class_list, np.arange(self.num_classes)))

        return np.vectorize(label_map.get)(label)
