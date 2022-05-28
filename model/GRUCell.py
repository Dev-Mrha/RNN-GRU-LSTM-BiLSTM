import torch.nn as nn
import torch


class GRUCell(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(GRUCell, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.linear1 = nn.Linear(input_dim, hidden_dim * 2)
        self.linear2 = nn.Linear(input_dim, hidden_dim)

    def forward(self, x, hidden_state):
        Batch_size, Node_num = x.size(0), x.size(1)
        hidden_state = hidden_state.reshape(
            (Batch_size, Node_num)
        )
        x_t = self.linear1(x)  # wzxt wrxt
        hidden_state_t1 = self.linear1(hidden_state)  # uzht-1 urht-1
        x1, x2 = torch.chunk(input=x_t, chunks=2, dim=1)  # devide to two
        hidden_state1, hidden_state2 = torch.chunk(input=hidden_state_t1, chunks=2, dim=1)

        z = torch.sigmoid(x1 + hidden_state1)  # z = sigmoid(wzxt + uzht-1 + bz)
        r = torch.sigmoid(x2 + hidden_state2)  # r = sigmoid(wrxt + urht-1 + br)

        x_t = self.linear2(x)  # whxt
        hidden_state_t1 = self.linear2(hidden_state)  # uhht-1

        h_hat = torch.tanh(x_t + r * hidden_state_t1)  # h_hat = tanh(whxt + uhht-1 * rt + bh)
        new_hidden_state = z * hidden_state + (1 - z) * h_hat  # ht = zt * ht-1 + (1 - zt) * h_hat
        return new_hidden_state, new_hidden_state


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class GRU(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(GRU, self).__init__()
        self.input_dim = input_dim  # num_nodes for prediction
        self.hidden_dim = hidden_dim
        self.gru_cell = GRUCell(self.input_dim, self.hidden_dim)

    def forward(self, data):
        x = data["flow_x"]  # x torch.Size([64, 307, 6, 1])
        x = x.to(device)
        Batch_size, Node_num = x.size(0), x.size(1)
        seq_len = x.size(2)
        x = x.permute(0, 2, 1, 3)
        x = x.view(Batch_size, seq_len, -1)  # x torch.Size([64, 6, 307])
        assert self.input_dim == Node_num
        outputs = list()
        hidden_state = torch.zeros(Batch_size, Node_num).type_as(x)
        # initial h0
        for i in range(seq_len):
            output, hidden_state = self.gru_cell(x[:, i, :], hidden_state)
            output = output.reshape((Batch_size, 1, Node_num))  # torch.Size([64, 307])
            outputs.append(output)
        last_output = outputs[-1]  # torch.Size([64, 1, 307])
        last_output = last_output.permute(0, 2, 1)  # torch.Size([64, 307, 1])
        last_output = last_output.reshape(Batch_size, Node_num, 1, 1)
        return last_output
