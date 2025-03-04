# model.py (MultiTarget Version)
import torch
import torch.nn as nn

class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size

        # Define query, key, and value transformations
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)

        self.register_buffer('scale', torch.sqrt(torch.FloatTensor([hidden_size])))

    def forward(self, x):
     # x shape: (batch, seq_len, hidden_size)
        Q = self.query(x)  # (batch, seq_len, hidden_size)
        K = self.key(x)    # (batch, seq_len, hidden_size)
        V = self.value(x)  # (batch, seq_len, hidden_size)

        # Scaled dot-product attention
        attention = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        attention = torch.softmax(attention, dim=-1)

        # Apply attention weights to values
        context = torch.matmul(attention, V)

        return context

class MultiTargetFinanceModel(nn.Module):
    def __init__(self, input_size_time_series, hidden_size_time_series, num_layers_time_series):
        """
        This model predicts five outputs: Open, High, Low, Close, and Volume.
        """
        super(MultiTargetFinanceModel, self).__init__()
        # Shared encoder
        self.lstm = nn.LSTM(input_size=input_size_time_series, hidden_size=hidden_size_time_series,
                            num_layers=num_layers_time_series, batch_first=True, dropout=0.2)
        self.attention = Attention(hidden_size_time_series)
        self.fc_time_series = nn.Sequential(
            nn.Linear(hidden_size_time_series, hidden_size_time_series // 2),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        # Separate prediction heads for each target variable
        self.head_open = nn.Sequential(
            nn.Linear(hidden_size_time_series // 2, hidden_size_time_series),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size_time_series, 1)
        )
        self.head_high = nn.Sequential(
            nn.Linear(hidden_size_time_series // 2, hidden_size_time_series),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size_time_series, 1)
        )
        self.head_low = nn.Sequential(
            nn.Linear(hidden_size_time_series // 2, hidden_size_time_series),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size_time_series, 1)
        )
        self.head_close = nn.Sequential(
            nn.Linear(hidden_size_time_series // 2, hidden_size_time_series),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size_time_series, 1)
        )
        self.head_volume = nn.Sequential(
            nn.Linear(hidden_size_time_series // 2, hidden_size_time_series),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size_time_series, 1)
        )

    def forward(self, x_ts, x_tab):
        lstm_out, _ = self.lstm(x_ts)                    # (batch, seq_len, hidden_size_time_series)
        ts_context = self.attention(lstm_out)
        ts_context = ts_context[:, -1, :]# (batch, hidden_size_time_series)
        ts_features = self.fc_time_series(ts_context)             # (batch, hidden_size_time_series//2)

        pred_open = self.head_open(ts_features)
        pred_high = self.head_high(ts_features)
        pred_low  = self.head_low(ts_features)
        pred_close = self.head_close(ts_features)
        pred_volume = self.head_volume(ts_features)

        # Concatenate outputs: (batch, 5)
        predictions = torch.cat([pred_open, pred_high, pred_low, pred_close, pred_volume], dim=1)
        return predictions
