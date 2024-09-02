import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class VaradeBlock(nn.Module):
    def __init__(self, filters, length):
        super(VaradeBlock, self).__init__()
        self.activation = nn.ReLU()
        self.dense = nn.Linear(length, length // 2)
        self.dense2 = nn.Linear(filters, filters)

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

class Varade(nn.Module):
    def __init__(self, filters=64, channels=1, inputLength=512):
        super(Varade, self).__init__()
        self.input_conv = nn.Conv1d(channels, filters, kernel_size=1, padding='same')
        self.nLayers = int(math.log(inputLength, 2))
        self.blocks = nn.ModuleList([VaradeBlock(filters, inputLength // 2**i) for i in range(self.nLayers)])
        self.out_mean = nn.Conv1d(filters, channels, kernel_size=1, padding='same')
        self.out_log_var = nn.Conv1d(filters, channels, kernel_size=1, padding='same')

    def forward(self, inputs):
        x=inputs
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

    def train_step(model, data, optimizer):
        x = data[:, :-1, :]
        y = data[:, -1:, :]
        model.train()
        optimizer.zero_grad()
        y_pred_mean, y_pred_log_var = model(x)
        y_pred_var = y_pred_log_var

        reconstruction_loss = torch.abs(y - y_pred_mean)
        error_loss = torch.abs(y_pred_var - reconstruction_loss)
        loss = reconstruction_loss + error_loss
        loss = loss.mean()

        loss.backward()
        optimizer.step()

        mse = F.mse_loss(y_pred_mean, y, reduction='mean')
        avgVar = y_pred_var.mean()

        return {'Log Likelihood': reconstruction_loss.mean().item(), 'MSE': mse.item(), 'Avg Variance': avgVar.item()}

