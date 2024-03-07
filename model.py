import torch
import torch.nn as nn


class GAT(nn.Module):

    def __init__(self, input_size, dropatt=0.0, **kwargs):
        super().__init__()

        # input projection
        self.W = nn.Parameter(torch.empty(input_size, input_size))

        # attention
        # [x1, x2] @ [a1, a2]' = x1 @ a1 + x2 @ a2
        self.a1 = nn.Parameter(torch.randn(1, input_size))
        self.a2 = nn.Parameter(torch.randn(input_size, 1))
        self.scale = (input_size * 2) ** 0.5

        self.leakyrelu = nn.LeakyReLU()
        self.dropatt = nn.Dropout(dropatt)

        # self.register_buffer("mask", - torch.eye(4096) * 1000)  # attention mask

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_normal_(self.a1)
        torch.nn.init.xavier_normal_(self.a2)
        torch.nn.init.eye_(self.W)

    def cal_attention(self, x):

        x1 = (x @ self.a1.squeeze() / self.scale).T[:, None, :]  # row vector
        x2 = (x @ self.a2.squeeze() / self.scale).T[:, :, None]  # column vector
        score = self.leakyrelu(x1 + x2)
        # score = score + self.mask[:len(x), :len(x)][None, :]  # don't use own features
        weight = torch.softmax(score, dim=-1)
        weight = self.dropatt(weight)  # attention dropout
        return weight

    def forward(self, x, return_attn=False):
        Wx = x @ self.W # batch * time * feature
        weight = self.cal_attention(Wx)  # time * batch * batch
        if return_attn:
            return weight
        agg = self.leakyrelu(weight @ Wx.permute(1, 0, 2)) # time * batch * feature
        agg = agg.permute(1, 0, 2)  # batch * time * feature
        return x - agg


class GRU(nn.Module):

    def __init__(self, input_size, hidden_size=64, num_layers=2, num_factors=5,
                 dropout=0.0, use_attn=True, **kwargs):
        super().__init__()

        self.rnn = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )

        self.use_attn = use_attn
        if self.use_attn:
            self.W = nn.Linear(hidden_size, hidden_size)
            self.u = nn.Linear(hidden_size, 1, bias=False)
            self.fc_out = nn.Linear(hidden_size * 2, num_factors)
        else:
            self.fc_out = nn.Linear(hidden_size, num_factors)

    def forward(self, x):
        rnn_out = self.rnn(x)[0]
        last_out = rnn_out[:, -1]
        if self.use_attn:
            laten = self.W(rnn_out).tanh()
            scores = self.u(laten).softmax(dim=1)
            att_out = (rnn_out * scores).sum(dim=1).squeeze()
            last_out = torch.cat([last_out, att_out], dim=1)
        return self.fc_out(last_out)


class Model(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

        num_factors = kwargs['num_factors']
        assert num_factors % 2 == 0
        kwargs['num_factors'] = num_factors // 2

        self.disable_gat = kwargs.get('disable_gat', False)
        if not self.disable_gat:
            self.gat = GAT(**kwargs)

        self.rnn1 = GRU(**kwargs)
        self.rnn2 = GRU(**kwargs)

        self.input_size = kwargs['input_size']

    def forward(self, x):
        # x: [N, F*T]
        x = x.reshape(len(x), self.input_size, -1) # [N, F, T]
        x = x.permute(0, 2, 1) # [N, T, F]
        if not self.disable_gat:
            x_agg = self.gat(x)
        else:
            x_agg = x
        return torch.cat([self.rnn1(x_agg), self.rnn2(x)], axis=1)
