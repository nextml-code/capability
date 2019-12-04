import torch
import torch.nn as nn
from torch.nn.functional import softmax


class TransformerClassifier(torch.nn.Module):

    def __init__(self, input_size, output_size):
        super().__init__()
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=input_size,
                nhead=5,
                dim_feedforward=input_size * 2
            ),
            num_layers=6
        )
        self.classifier = AttentionClassifier(input_size, output_size)

    def forward(self, x):
        x = self.encoder(x.permute(1, 0, 2))
        return self.classifier(x.permute(1, 0, 2))


class AttentionClassifier(torch.nn.Module):

    def __init__(self, input_size, output_size):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=64,
            num_layers=1,
            bidirectional=True,
            batch_first=True
        )
        lstm_output_size = self.lstm.hidden_size * (1 + self.lstm.bidirectional)
        self.attention_weights = nn.Linear(
            lstm_output_size,
            lstm_output_size,
            bias=False
        )
        self.dropout = nn.Dropout(p=0.6)
        self.classifier = nn.Linear(lstm_output_size, output_size)

    def attention_layer(self, lstm_output, hidden):
        hidden = hidden.view(
            self.lstm.num_layers, (1 + self.lstm.bidirectional),
            hidden.size(1), hidden.size(2)
        )
        hidden = torch.cat([*hidden[-1]], dim=1)
        lstm_output = self.attention_weights(lstm_output)
        attn_weights = torch.bmm(lstm_output, hidden.unsqueeze(2)).squeeze(2)
        soft_attn_weights = softmax(attn_weights, 1)
        new_hidden_state = torch.bmm(
            lstm_output.transpose(1, 2), soft_attn_weights.unsqueeze(2)
        )
        return new_hidden_state.squeeze(2)

    def forward(self, x):
        x, (h_n, c_n) = self.lstm(x)
        x = self.attention_layer(x, h_n)
        x = self.classifier(self.dropout(x))
        return x.squeeze(1)
