from typing import List, Union, Callable, Tuple

import torch
from torch.nn import functional as F
from tqdm import tqdm

from ..common import RNN, MLP, PredictionAnomalyDetector
from ...models import BaseModel
from ...data.transforms import PredictionTargetTransform, Transform
from ...utils import torch_utils
from ...utils.utils import halflife2alpha
from ...optim.loss import Loss

class Naive(BaseModel):
    def __init__(self):
        super(Naive, self).__init__()

    def forward(self, inputs):
        return torch.zeros_like(inputs[0][:,0:1,:])


class NaiveAnomalyDetector(PredictionAnomalyDetector):
    def __init__(self, model: Naive):
        super(NaiveAnomalyDetector, self).__init__()
        self.model = model

    def fit(self, dataset: torch.utils.data.DataLoader) -> None:
        errors = []
        mean = 0
        total = 0

        # Compute mean and covariance over the entire validation dataset
        for i, (b_inputs, b_targets) in tqdm(enumerate(dataset)):
            b_inputs = tuple(b_inp for b_inp in b_inputs)
            b_targets = tuple(b_tar for b_tar in b_targets)
            with torch.no_grad():
                pred = self.model(b_inputs)

            target, = b_targets

            if i == 0:
                # Use all datapoints in the first window
                error = target - pred
            else:
                # In all subsequent windows, only the last datapoint will be new
                error = target[-1:] - pred[-1:]
            error.abs_()
            errors.append(error.reshape(-1, error.shape[-1]))

            mean += torch.sum(error, dim=(0, 1))
            total += error.shape[0] * error.shape[1]

            if i>10000:
                break

        mean /= total

        errors = torch.cat(errors, dim=0)
        errors -= mean
        cov = torch.matmul(errors.T, errors)
        cov /= total - 1

        # Add a small epsilon to the diagonal of the matrix to make it non-singular
        cov.diagonal().add_(1e-5)

        # This construction ensures that the resulting precision matrix is pos. semi-definite, even if the condition
        # number of the cov matrix is large
        cholesky = torch.linalg.cholesky(cov)
        precision = cov
        torch.cholesky_inverse(cholesky, out=precision)

        self.register_buffer('mean', mean)
        self.register_buffer('precision', precision)

    def compute_online_anomaly_score(self, inputs: Tuple[torch.Tensor, torch.Tensor, float, float]) \
            -> Tuple[torch.Tensor, float, float]:
        # x: (T, B, D), target: (T, B, D), moving_avg: ()
        x, target = inputs

        with torch.no_grad():
            x_pred = self.model((x,))

        error = torch.abs(target - x_pred)
        error -= self.mean

        result = F.bilinear(error, error, self.precision.unsqueeze(0))
        return result.squeeze(-1)

    def compute_offline_anomaly_score(self, inputs: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        raise NotImplementedError

    def format_online_targets(self, targets: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        pass

    def get_labels_and_scores(self, dataset: torch.utils.data.DataLoader) -> Tuple[torch.Tensor, torch.Tensor]:
        errors = []
        labels = []
        moving_avg_num = 0
        moving_avg_denom = 0

        # Compute exp moving average of error score
        for b_inputs, b_targets in dataset:
            b_inputs = tuple(b_inp.to(self.dummy.device) for b_inp in b_inputs)
            b_targets = tuple(b_tar.to(self.dummy.device) for b_tar in b_targets)

            x, = b_inputs
            label, target = b_targets

            sq_error = self.compute_online_anomaly_score((x, target))
            errors.append(sq_error)
            labels.append(label.cpu())

        scores = torch.cat(errors, dim=1).transpose(0, 1).flatten()
        labels = torch.cat(labels, dim=1).transpose(0, 1).flatten()

        assert labels.shape == scores.shape

        return labels, scores.cpu()


class LSTMS2STargetTransform(PredictionTargetTransform):
    def __init__(self, parent: Transform, window_size: int, replace_labels: bool = False,
                 reverse: bool = False):
        super(LSTMS2STargetTransform, self).__init__(parent, window_size, window_size, replace_labels=replace_labels,
                                                     step_size=window_size, reverse=reverse)