from typing import List, Union, Callable, Tuple

import torch
from torch.nn import functional as F

from ..common import RNN, MLP, PredictionAnomalyDetector
from ...models import BaseModel
from ...data.transforms import PredictionTargetTransform, Transform
from ...utils import torch_utils
from ...utils.utils import halflife2alpha
from ...optim.loss import Loss

class VaradeBlock(torch.nn.Module):
    def __init__(self, filters, length):
        super(VaradeBlock, self).__init__()
        self.activation = torch.nn.ReLU()
        self.dense = torch.nn.Linear(length, length // 2)
        self.dense2 = torch.nn.Linear(filters, filters)

    def positionalEncoding(self, inputs):
        _, seq_length, _ = inputs.size()
        x = torch.arange(seq_length, dtype=torch.float32, device=inputs.device)
        x = x.view(1, -1, 1) / seq_length
        x = x.expand(inputs.size(0), -1, -1)
        return torch.cat([inputs, x], dim=-1)

    def forward(self, inputs):
        x = self.activation(inputs)
        #x = self.positionalEncoding(x)
        x = x.transpose(1, 2)
        x = self.dense(x)
        x = x.transpose(1, 2)
        x = self.dense2(x)
        return x

class Varade(BaseModel):
    def __init__(self, filters=64, channels=1, inputLength=512):
        super(Varade, self).__init__()
        self.input_conv = torch.nn.Conv1d(channels, filters, kernel_size=1, padding='same')
        self.nLayers = int(torch.math.log(inputLength, 2))
        self.blocks = torch.nn.ModuleList([VaradeBlock(filters, inputLength // 2**i) for i in range(self.nLayers)])
        self.out_mean = torch.nn.Conv1d(filters, channels, kernel_size=1, padding='same')
        self.out_log_var = torch.nn.Conv1d(filters, channels, kernel_size=1, padding='same')

    def forward(self, inputs):
        x=inputs[0]
        x=x.transpose(1,2)
        x = self.input_conv(x)
        x=x.transpose(1,2)

        for block in self.blocks:
            x = block(x)

        x=x.transpose(1,2)
        out_diff = self.out_mean(x).transpose(1,2)
        out_log_var = F.relu(self.out_log_var(x)) + 1e-2
        out_log_var = out_log_var.transpose(1,2)

        return out_diff, out_log_var


class VaradeAnomalyDetector(PredictionAnomalyDetector):
    def __init__(self, model: Varade):
        super(VaradeAnomalyDetector, self).__init__()
        self.model = model


    def fit(self, dataset: torch.utils.data.DataLoader) -> None:
        pass

    def compute_online_anomaly_score(self, inputs: Tuple[torch.Tensor, torch.Tensor, float, float]) \
            -> Tuple[torch.Tensor, float, float]:
        # x: (T, B, D), target: (T, B, D), moving_avg: ()
        x, target = inputs

        with torch.no_grad():
            x_pred,x_logvar = self.model((x,))

        error = torch.abs(target - x_pred)/x_logvar
        error=torch.norm(error,dim=-1)

        T, B = error.shape
        sq_error = error.T.flatten()

        return error.view(B, T).T

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

class VaradeLoss(Loss):
    def __init__(self, model: Varade):
        super(VaradeLoss, self).__init__()
        self.model = model

    def forward(
        self,
        predictions: Tuple[torch.Tensor, ...],
        targets: Tuple[torch.Tensor, ...] = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:

        targets=targets[0]
        y_pred_mean, y_pred_log_var = predictions
        y_pred_var = y_pred_log_var

        reconstruction_loss = torch.abs(targets - y_pred_mean)
        error_loss = torch.abs(y_pred_var - reconstruction_loss)
        loss = reconstruction_loss + error_loss
        loss = loss.mean()
        return loss