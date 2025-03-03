# model.py (MultiTarget Version)
import torch
import torch.nn as nn
import torch.optim as optim
import data_loader

class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size

        # Define query, key, and value transformations
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)

        self.scale = torch.sqrt(torch.FloatTensor([hidden_size]))

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
    def __init__(self, input_size_time_series, hidden_size_time_series, num_layers_time_series,
                 input_size_tab, hidden_size_tab):
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
        self.fc_tab = nn.Sequential(
            nn.Linear(input_size_tab, hidden_size_tab),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        combined_input_size = hidden_size_time_series // 2 + hidden_size_tab

        # Separate prediction heads for each target variable
        self.head_open = nn.Sequential(
            nn.Linear(combined_input_size, hidden_size_time_series),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size_time_series, 1)
        )
        self.head_high = nn.Sequential(
            nn.Linear(combined_input_size, hidden_size_time_series),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size_time_series, 1)
        )
        self.head_low = nn.Sequential(
            nn.Linear(combined_input_size, hidden_size_time_series),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size_time_series, 1)
        )
        self.head_close = nn.Sequential(
            nn.Linear(combined_input_size, hidden_size_time_series),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size_time_series, 1)
        )
        self.head_volume = nn.Sequential(
            nn.Linear(combined_input_size, hidden_size_time_series),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size_time_series, 1)
        )

    def forward(self, x_ts, x_tab):
        lstm_out, _ = self.lstm(x_ts)                    # (batch, seq_len, hidden_size_time_series)
        ts_context = self.attention(lstm_out)
        ts_context = ts_context[:, -1, :]# (batch, hidden_size_time_series)
        ts_features = self.fc_time_series(ts_context)             # (batch, hidden_size_time_series//2)
        tab_features = self.fc_tab(x_tab)                # (batch, hidden_size_tab)
        combined_features = torch.cat([ts_features, tab_features], dim=1)

        pred_open = self.head_open(combined_features)
        pred_high = self.head_high(combined_features)
        pred_low  = self.head_low(combined_features)
        pred_close = self.head_close(combined_features)
        pred_volume = self.head_volume(combined_features)

        # Concatenate outputs: (batch, 5)
        predictions = torch.cat([pred_open, pred_high, pred_low, pred_close, pred_volume], dim=1)
        return predictions

def train_model(model, train_loader, num_epochs=200, device=torch.device("cpu")):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.9)

    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for x_ts, x_tab, y in train_loader:
            x_ts, x_tab, y = x_ts.to(device), x_tab.to(device), y.to(device)
            optimizer.zero_grad()
            outputs = model(x_ts, x_tab)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        scheduler.step()
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss/len(train_loader):.6f}")

    save_path = "multi_target_finance_model.pth"
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")
