import torch
import torch.nn as nn


class CNNBiLSTM(nn.Module):
    def __init__(
        self,
        input_size: int,
        cnn_out_channels: int,
        cnn_kernel_size: int,
        cnn_stride: int,
        cnn_pool_size: int,
        hidden_size: int,
        num_lstm_layers: int,
        dense_size: int,
    ):
        super().__init__()

        self.conv1 = nn.Conv1d(
            input_size,
            cnn_out_channels,
            kernel_size=cnn_kernel_size,
            stride=cnn_stride,
            padding=cnn_kernel_size // 2,
        )

        self.pool = nn.Identity() if cnn_pool_size <= 1 else nn.MaxPool1d(cnn_pool_size)

        self.lstm = nn.LSTM(
            cnn_out_channels,
            hidden_size,
            num_layers=num_lstm_layers,
            batch_first=True,
            bidirectional=True,
        )

        self.fc1 = nn.Linear(hidden_size * 2, dense_size)
        self.out = nn.Linear(dense_size, 1)

    def forward(self, x):
        # x: [B, T, F]
        x = x.transpose(1, 2)  # [B, F, T]
        x = torch.relu(self.conv1(x))
        x = self.pool(x)
        x = x.transpose(1, 2)  # [B, T, C]

        lstm_out, _ = self.lstm(x)
        h = lstm_out[:, -1, :]  # last time step

        h = torch.relu(self.fc1(h))
        out = self.out(h).squeeze(-1)

        return out
