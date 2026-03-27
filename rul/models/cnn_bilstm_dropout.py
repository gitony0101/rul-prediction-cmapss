import torch
import torch.nn as nn


class CNNBiLSTMDropout(nn.Module):
    def __init__(
        self,
        input_size,
        cnn_out_channels,
        cnn_kernel_size,
        cnn_stride,
        cnn_pool_size,
        hidden_size,
        num_lstm_layers,
        dense_size,
        dropout_rate,
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

        self.drop1 = nn.Dropout(dropout_rate)
        self.drop2 = nn.Dropout(dropout_rate)

        self.fc1 = nn.Linear(hidden_size * 2, dense_size)
        self.out = nn.Linear(dense_size, 1)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = torch.relu(self.conv1(x))
        x = self.pool(x)
        x = x.transpose(1, 2)

        lstm_out, _ = self.lstm(x)
        h = lstm_out[:, -1, :]

        h = self.drop1(h)
        h = torch.relu(self.fc1(h))
        h = self.drop2(h)

        return self.out(h).squeeze(-1)
