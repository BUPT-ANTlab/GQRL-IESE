import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy
from os.path import join
import math
import matplotlib.pyplot as plt


class PursuitModule(torch.nn.Module):
    def __init__(self, input_ego_state_shape, input_oth_state_shape,input_eva_state_shape, outputs, max_history_length, network_constructor):
        super(PursuitModule, self).__init__()
        self.protagonist_net = network_constructor(input_ego_state_shape, input_oth_state_shape,input_eva_state_shape, outputs, max_history_length).to("cuda")
    def protagonist_parameters(self):
        return self.protagonist_net.parameters()

    def forward(self, x, hidden_state=None, use_gumbel_softmax=False):
        if hidden_state is None:
            if use_gumbel_softmax:
                return self.protagonist_net(x, use_gumbel_softmax=True)
            return self.protagonist_net(x)
        else:
            return self.protagonist_net(x, hidden_state)

    def save_weights(self, path):
        protagonist_path = join(path, "pursuit_model.pth")
        torch.save(self.protagonist_net.state_dict(), protagonist_path)

    def load_weights(self, path):
        protagonist_path = join(path, "pursuit_model.pth")
        # self.protagonist_net.load_state_dict(torch.load(protagonist_path, map_location='cpu'))
        self.protagonist_net.load_state_dict(torch.load(protagonist_path, map_location='cuda'))
        self.protagonist_net.eval()

    def load_weights_from_history(self, path):
        # self.protagonist_net = torch.load(path, map_location='cuda')
        self.protagonist_net.load_state_dict(torch.load(path, map_location='cuda'))
        self.protagonist_net.eval()

    def save_weights_to_path(self, path):
        torch.save(self.protagonist_net.state_dict(), path)


class IESE(nn.Module):
    def __init__(self, input_ego_state_shape, input_oth_state_shape,input_eva_state_shape, max_sequence_length, nr_hidden_units=8, params=None):
        super(IESE, self).__init__()
        self.input_ego_state_shape=input_ego_state_shape
        self.input_oth_state_shape=input_oth_state_shape
        self.input_eva_state_shape=input_eva_state_shape
        self.nr_hidden_units=nr_hidden_units
        self.conv_oth_net = nn.Sequential(
            nn.Conv1d(self.input_oth_state_shape[0], 2, kernel_size=4),
            nn.ELU(),
            nn.Flatten()
        )#1*14
        self.attention_weight = nn.Sequential(
            nn.Conv1d(2, 1, kernel_size=1, stride=1, padding=0, bias=True),

            nn.BatchNorm1d(1),
            nn.Sigmoid()  # Sigmoid
        )
        self.relu = nn.ReLU(inplace=True)
        self.changeshape1 = nn.Sequential(
            nn.Flatten()
        )
        self.changeshape2 = nn.Sequential(
            nn.Flatten()
        )
        self.W_ego = nn.Sequential(
            nn.Conv1d(8, 2, kernel_size=1, stride=1, padding=0, bias=True),  # 1*1�����F_int�Ǹò������ͨ����
            nn.BatchNorm1d(2)
        )#1*2*10
        self.W_eva = nn.Sequential(
            nn.Conv1d(8, 2, kernel_size=1, stride=1, padding=0, bias=True),  # 1*1�����F_int�Ǹò������ͨ����
            nn.BatchNorm1d(2)
        )#1*2*10
        self.fc_net = nn.Sequential(
            nn.Linear(174, 64),
            nn.ELU(),
            nn.Linear(64, 32),
            nn.ELU(),
            nn.Linear(32, nr_hidden_units),
            nn.ELU()
        )
    def forward(self, sin_obs):
        single_obs = np.swapaxes(np.array(sin_obs), 0, 1)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ego_state = torch.tensor(single_obs[0], device=self.device, dtype=torch.float32)
        oth_state = torch.tensor(single_obs[1], device=self.device, dtype=torch.float32)
        eva_state = torch.tensor(single_obs[2], device=self.device, dtype=torch.float32)
        # conv_ego_state = self.conv_ego_net(ego_state)  # 1*4*9
        coded_oth_state = self.conv_oth_net(oth_state) #1*14
        ego_state_att = self.W_ego(ego_state) ##1*2*10
        eva_state_att = self.W_eva(eva_state) ##1*2*10
        att_weight = self.relu(ego_state_att + eva_state_att)
        att_weight = self.attention_weight(att_weight) #1*1*10
        att_state = eva_state * att_weight #1*8*10
        coded_att_state = self.changeshape1(att_state) #80
        coded_ego_state = self.changeshape2(ego_state) #80
        coded_info = torch.cat((coded_ego_state, coded_oth_state), -1)
        coded_info = torch.cat((coded_info, coded_att_state), -1)
        return self.fc_net(coded_info)


class MLPHidden(nn.Module):

    def __init__(self, input_shape, max_sequence_length, nr_hidden_units=64):
        super(MLPHidden, self).__init__()
        self.input_shape = input_shape
        self.nr_input_features = numpy.prod(self.input_shape)*max_sequence_length
        self.max_sequence_length = max_sequence_length
        self.nr_hidden_units = nr_hidden_units
        if max_sequence_length > 1:
            self.nr_hidden_units = int(2*nr_hidden_units)
        self.fc_net = nn.Sequential(
            nn.Linear(self.nr_input_features, self.nr_hidden_units),
            nn.ELU(),
            nn.Linear(self.nr_hidden_units, self.nr_hidden_units),
            nn.ELU()
        )

    def forward(self, x, h):
        sequence_length = x.size(0)
        batch_size = x.size(1)
        assert self.max_sequence_length == sequence_length, "Got shape: {}".format(x.size())
        x = x.view(sequence_length, batch_size, -1)
        x = x.permute(1, 0, 2)
        x = torch.reshape(x, (batch_size, -1))
        return self.fc_net(x)


class MLP3D(nn.Module):
    def __init__(self, input_shape, max_sequence_length, nr_hidden_units=64):
        super(MLP3D, self).__init__()
        self.input_shape = input_shape
        # self.nr_input_features = numpy.prod(self.input_shape) * max_sequence_length
        self.nr_input_features = 640
        self.max_sequence_length = max_sequence_length
        self.nr_hidden_units = nr_hidden_units
        if max_sequence_length > 1:
            self.nr_hidden_units = int(2 * nr_hidden_units)
        self.fc_net = nn.Sequential(
            nn.Linear(self.nr_input_features, self.nr_hidden_units),
            nn.ELU(),
            nn.Linear(self.nr_hidden_units, self.nr_hidden_units),
            nn.ELU()
        )

    def forward(self, x):
        sequence_length = x.size(0)
        batch_size = x.size(1)
        assert self.max_sequence_length == sequence_length, "Got shape: {}".format(x.size())
        # x = x.view(sequence_length, batch_size, -1)
        x = x.permute(1, 2, 0, 3, 4)
        m_3d = torch.nn.Conv3d(10, 20, 2, stride=1).to("cuda")
        x = m_3d(x)
        x = torch.reshape(x, (batch_size, -1))
        return self.fc_net(x)


class TimeTransformer(nn.Module):
    def __init__(self, input_shape, max_sequence_length, params, q_value):
        super(TimeTransformer, self).__init__()
        self.q_values = q_value
        self.params = params
        self.input_shape = input_shape
        self.max_sequence_length = max_sequence_length
        self.transformer = TTransformer(self.params["token_dim"], self.params["emb"], self.params["heads"],
                                        self.params["depth"], self.params["emb"])
        self.q_basic = nn.Linear(self.params["emb"], self.params["nr_actions"])
        if self.q_values:
            self.value_head = nn.Linear(self.params["emb"], self.params["nr_actions"])
        else:
            self.value_head = nn.Linear(self.params["emb"], 1)
        self.te = self.TimeEmbedding(max_sequence_length, params["token_dim"])
        self.te = self.te.view(max_sequence_length, -1, params["token_dim"])

    def forward(self, x):
        sequence_length = x.size(0)
        batch_size = x.size(1)
        assert self.max_sequence_length == sequence_length, "Got shape: {}".format(x.size())
        x = x.view(sequence_length, batch_size, -1)
        x = x + self.te
        outputs = self.transformer.forward(x, None)
        q_basic_actions = self.q_basic(outputs)
        q = q_basic_actions
        value = self.value_head(outputs)
        return q.mean(0), value.mean(0)

    def TimeEmbedding(self,max_sequence_length, d_model):
        self.max_sequence_length = max_sequence_length
        te = torch.zeros(self.max_sequence_length, d_model, device="cuda")
        position = torch.arange(0, self.max_sequence_length).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))
        te[:, 0::2] = torch.sin(position * div_term)
        te[:, 1::2] = torch.cos(position * div_term)
        te = te.unsqueeze(0)
        return te


class UPDeT(nn.Module):
    def __init__(self, input_shape, params):
        super(UPDeT, self).__init__()
        self.params = params
        self.transformer = Transformer(self.params["token_dim"], self.params["emb"], self.params["heads"],
                                       self.params["depth"], self.params["emb"])
        self.q_basic = nn.Linear(self.params["emb"], self.params["nr_actions"])

        if params["decoupling"]:
            self.go_basic = nn.Linear(self.params["emb"], self.params["nr_actions"])

    def init_hidden(self):
        # make hidden states on same device as model
        return torch.zeros(1, self.params["emb"]).cuda()

    def forward(self, inputs, hidden_state, task_enemy_num, task_ally_num):
        if self.params["decoupling"]:
            outputs, _ = self.transformer.forward(inputs, hidden_state, None, True)
            if self.params["local_observation_format"] == 0:
                go_value_self = self.go_basic(outputs[:, 0, :])
                go_value_obs = self.go_basic(outputs[:, -1, :])
                go_value = go_value_obs + go_value_self


                q_value = self.q_basic(outputs[:, 1, :])
                for i in range(2, 9):
                    q_value += self.q_basic(outputs[:, i, :])
                q_value = q_value/8

                q_value = q_value + go_value

                q = q_value
                h = outputs[:, -1:, :]
                return q, h
            else:
                go_value_self = self.go_basic(outputs[:, 0, :])
                go_value_obs = self.go_basic(outputs[:, 1, :])
                go_value = go_value_self + go_value_obs
                q_value = self.q_basic(outputs[:, 2, :]) + go_value

                q = q_value
                h = outputs[:, -1:, :]
                return q, h

        else:
            outputs, _ = self.transformer.forward(inputs, hidden_state, None, False)
            # first output for 6 action (no_op stop up down left right)
            q_basic_actions = self.q_basic(outputs[:, 0, :])

            # last dim for hidden state
            h = outputs[:, -1:, :]

            q = q_basic_actions
            return q, h


class SelfAttention(nn.Module):
    def __init__(self, emb, heads=8, mask=False):

        super().__init__()
        self.emb = emb
        self.heads = heads
        self.mask = mask
        self.tokeys = nn.Linear(emb, emb * heads, bias=False)
        self.toqueries = nn.Linear(emb, emb * heads, bias=False)
        self.tovalues = nn.Linear(emb, emb * heads, bias=False)
        self.unifyheads = nn.Linear(heads * emb, emb)
    def forward(self, x, mask):
        b, t, e = x.size()
        h = self.heads
        keys = self.tokeys(x).view(b, t, h, e)
        queries = self.toqueries(x).view(b, t, h, e)
        values = self.tovalues(x).view(b, t, h, e)
        keys = keys.transpose(1, 2).contiguous().view(b * h, t, e)
        queries = queries.transpose(1, 2).contiguous().view(b * h, t, e)
        values = values.transpose(1, 2).contiguous().view(b * h, t, e)
        queries = queries / (e ** (1 / 4))
        keys = keys / (e ** (1 / 4))
        dot = torch.bmm(queries, keys.transpose(1, 2))
        assert dot.size() == (b * h, t, t)
        if self.mask:  # mask out the upper half of the dot matrix, excluding the diagonal
            mask_(dot, maskval=float('-inf'), mask_diagonal=False)
        if mask is not None:
            dot = dot.masked_fill(mask == 0, -1e9)
        dot = F.softmax(dot, dim=2)
        out = torch.bmm(dot, values).view(b, h, t, e)
        out = out.transpose(1, 2).contiguous().view(b, t, h * e)
        return self.unifyheads(out)


class TransformerBlock(nn.Module):

    def __init__(self, emb, heads, mask, ff_hidden_mult=4, dropout=0.0):
        super().__init__()

        self.attention = SelfAttention(emb, heads=heads, mask=mask)
        self.mask = mask

        self.norm1 = nn.LayerNorm(emb)
        self.norm2 = nn.LayerNorm(emb)

        self.ff = nn.Sequential(
            nn.Linear(emb, ff_hidden_mult * emb),
            nn.ReLU(),
            nn.Linear(ff_hidden_mult * emb, emb)
        )

        self.do = nn.Dropout(dropout)

    def forward(self, x_mask):
        x, mask = x_mask

        attended = self.attention(x, mask)

        x = self.norm1(attended + x)

        x = self.do(x)

        fedforward = self.ff(x)

        x = self.norm2(fedforward + x)

        x = self.do(x)

        return x, mask


class Transformer(nn.Module):

    def __init__(self, input_dim, emb, heads, depth, output_dim):
        super().__init__()

        self.num_tokens = output_dim

        self.token_embedding = nn.Linear(input_dim, emb)

        tblocks = []
        for i in range(depth):
            tblocks.append(
                TransformerBlock(emb=emb, heads=heads, mask=False))

        self.tblocks = nn.Sequential(*tblocks)

        self.toprobs = nn.Linear(emb, output_dim)

    def forward(self, x, h, mask, decoupling):
        # sequence_length = x.size(0)
        # batch_size = x.size(1)
        # x = x.view(sequence_length, batch_size, -1).view(sequence_length, batch_size, 1, -1)
        # # x = x.permute(1, 0, 2)
        # tokens = self.token_embedding(x)
        # tokens = torch.cat((tokens, h), 1)
        if decoupling:
            x_batch = x.size(0)
            x_t = x.size(1)
            num_decoupling = x.size(2)
            x = x.contiguous().view(x_batch * x_t, num_decoupling, -1)
            h_size = h.size(0)
            h_t = h.size(1)
            h = h.contiguous().view(h_size * h_t, 1, -1)
            tokens = self.token_embedding(x)
            tokens = torch.cat((tokens, h), 1)

            b, t, e = tokens.size()

            x, mask = self.tblocks((tokens, mask))

            x = self.toprobs(x.view(b * t, e)).view(b, t, self.num_tokens)

            return x, tokens
        else:
            x_batch = x.size(0)
            x_t = x.size(1)
            x = x.contiguous().view(x_batch*x_t, 1, -1)
            h_size = h.size(0)
            h_t = h.size(1)
            h = h.contiguous().view(h_size*h_t, 1, -1)
            tokens = self.token_embedding(x)
            tokens = torch.cat((tokens, h), 1)

            b, t, e = tokens.size()

            x, mask = self.tblocks((tokens, mask))

            x = self.toprobs(x.view(b * t, e)).view(b, t, self.num_tokens)

            return x, tokens


class TTransformer(nn.Module):
    def __init__(self, input_dim, emb, heads, depth, output_dim):
        super().__init__()

        self.num_tokens = output_dim

        self.token_embedding = nn.Linear(input_dim, emb)

        tblocks = []
        for i in range(depth):
            tblocks.append(
                TransformerBlock(emb=emb, heads=heads, mask=False))

        self.tblocks = nn.Sequential(*tblocks)

        self.toprobs = nn.Linear(emb, output_dim)

    def forward(self, x, mask):

        x_batch = x.size(0)
        x_t = x.size(1)
        x = x.view(x_batch * x_t, 1, -1)
        tokens = self.token_embedding(x)

        b, t, e = tokens.size()

        x, mask = self.tblocks((tokens, mask))

        x = self.toprobs(x.view(b * t, e)).view(b, t, self.num_tokens)

        return x


def mask_(matrices, maskval=0.0, mask_diagonal=True):

    b, h, w = matrices.size()
    indices = torch.triu_indices(h, w, offset=0 if mask_diagonal else 1)
    matrices[:, indices[0], indices[1]] = maskval
