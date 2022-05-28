import torch
import torch.nn as nn
from GRUCell import *


class GRURnn(nn.Module):
    def __init__(self):
        super(GRURnn, self).__init__()
        self.embedding = nn.Embedding(inputs_dim, 256)
        self.gru = GRU(256, 256)
        self.linear = nn.Linear(256 * 2, out_features=10)
        self.dropout = nn.Dropout(0.5)

    def forward(self, inputs):
        emb = self.dropout(self.embedding(inputs))
        output, hidden = self.gru(emb)
        hidden = self.dropout(hidden)
        return self.linear(hidden)
