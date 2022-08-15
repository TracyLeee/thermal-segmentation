import numpy as np


class ConfusionMatrix:
    """This class computes the confusion matrix given true and prediction annotations.
    Check [confusion matrix](https://en.wikipedia.org/wiki/Confusion_matrix).
    """

    def __init__(self, labels: list[int]) -> None:
        """
        :param labels: a list of label ids used in this dataset
        """
        self._labels = labels
        self._label_num = len(self._labels)
        self._label_idx = np.arange(self._label_num)
        self._label_to_idx = dict(zip(self._labels, self._label_idx))
        self._matrix = np.zeros((self._label_num, self._label_num), dtype=np.uint64)

    def reset(self) -> None:
        """Resets the confusion matrix to be zero matrix"""
        self._matrix = np.zeros((self._label_num, self._label_num))

    def update(self, true_annot: np.ndarray, pred_annot: np.ndarray) -> None:
        """Updates the confusion matrix given true and predicted annotations

        :param true_annot: true annotations -- 2-D numpy array (image) of label ids
        :param pred_annot: predicted annotations -- 2-D numpy array (image) of label ids
        """
        # Convert values into range [0, self.lbl_num)
        true_annot = np.vectorize(self._label_to_idx.get)(true_annot).ravel()
        pred_annot = np.vectorize(self._label_to_idx.get)(pred_annot).ravel()

        matrix = (
            np.bincount(
                true_annot * self._label_num + pred_annot,
                minlength=self._label_num**2,
            )
            .reshape(self._label_num, self._label_num)
            .astype(np.uint64)
        )

        self._matrix += matrix

    @property
    def confusion_matrix(self) -> np.ndarray:
        """Returns the confusion matrix, whose shape is (self._label_num, self._label_num).
        The element at (row, col) denotes the number of pixels, which are in {row}th
        category in true annotations but classified as {col}th category in predicted
        annotations.
        """
        return self._matrix

    @property
    def iou(self) -> np.ndarray:
        """This class also computes intersection over union (IoU, a.k.a. Jaccard index).
        Overall IoU of all categories, as well as IoU of each category is computed.
        Check [Jaccard index](https://en.wikipedia.org/wiki/Jaccard_index).

        Returns IoU of each category"""
        return np.diag(self._matrix) / (
            np.sum(self._matrix, axis=1)  # sum per category in true annotations
            + np.sum(self._matrix, axis=0)  # sum per category in predicted annotations
            - np.diag(self._matrix)
        )

    @property
    def overall_iou(self) -> float:
        """
        Overall IoU treats all different categories as a single one.
        So overall IoU is computed as:
        {number of correctly classified pixels} / (
            {number of mis-classified pixels in two images}
            + {number of correctly classified pixels}
        )

        Returns IoU of all categories"""
        return np.sum(np.diag(self._matrix)) / (
            np.sum(self._matrix) * 2 - np.sum(np.diag(self._matrix))
        )

    @property
    def acc(self) -> np.ndarray:
        return np.diag(self._matrix) / np.sum(self._matrix, axis=1)

    @property
    def overall_acc(self) -> float:
        return np.sum(np.diag(self._matrix)) / np.sum(self._matrix)


class SegmentMetrics(object):
    def __init__(self, labels: list[int]) -> None:
        self.labels = labels
        self.matrix = ConfusionMatrix(self.labels)
        self.loss = 0.0
        self.count = 0

    def update(
        self,
        true_annot: np.ndarray,
        pred_annot: np.ndarray,
        loss: float,
        batch_size: int,
    ) -> None:
        if len(true_annot.shape) != 3:
            raise ValueError("Invalid dimension of labels!")

        if true_annot.shape[0] != batch_size:
            raise ValueError(
                f"Labels batch size: {true_annot.shape[0]}, batch size given: {batch_size}"
            )

        for idx in range(batch_size):
            self.matrix.update(true_annot[idx, ...], pred_annot[idx, ...])

        self.loss += loss
        self.count += batch_size

    def reset(self) -> None:
        self.matrix.reset()
        self.loss = 0.0
        self.count = 0

    @property
    def avg_loss(self) -> float:
        return self.loss / self.count

    @property
    def total_loss(self) -> float:
        return self.loss

    @property
    def iou(self) -> np.ndarray:
        return self.matrix.iou

    @property
    def overall_iou(self) -> float:
        return self.matrix.overall_iou

    @property
    def acc(self) -> np.ndarray:
        return self.matrix.acc

    @property
    def overall_acc(self) -> float:
        return self.matrix.overall_acc


class SegmentMetricLoss(object):
    def __init__(self) -> None:
        self.loss = 0.0
        self.count = 0

    def update(self, loss: float, batch_size: int) -> None:
        self.loss += loss
        self.count += batch_size

    def reset(self) -> None:
        self.loss = 0.0
        self.count = 0

    @property
    def avg_loss(self) -> float:
        return self.loss / self.count

    @property
    def total_loss(self) -> float:
        return self.loss
