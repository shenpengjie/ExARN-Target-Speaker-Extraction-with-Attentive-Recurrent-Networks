import math

import torch
from torch import nn

class attn_block(nn.Module):
    def __init__(self, feature_dim, infer=False, causal=True):
        super(attn_block, self).__init__()
        self.feature_dim = feature_dim
        self.infer = infer
        self.causal = causal
        # self.query_vector = nn.Parameter(torch.FloatTensor(self.feature_dim, ), requires_grad=True)
        # self.key_vector = nn.Parameter(torch.FloatTensor(self.feature_dim, ), requires_grad=True)
        # self.value_vector = nn.Parameter(torch.FloatTensor(self.feature_dim), requires_grad=True)
        self.query_linear = nn.Linear(self.feature_dim, self.feature_dim)
        self.value_linear_sig = nn.Linear(self.feature_dim, self.feature_dim)
        self.value_linear_tan = nn.Linear(self.feature_dim, self.feature_dim)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=-1)
        # self.reset_parameters()

    # def reset_parameters(self):
    #     # torch.nn.init.xavier_normal_(self.query_vector)
    #     # torch.nn.init.xavier_normal_(self.key_vector)
    #     # torch.nn.init.xavier_normal_(self.value_vector)
    #     stdv = 1. / math.sqrt(self.query_vector.size(0))
    #     self.query_vector.data.uniform_(-stdv, stdv)
    #     self.key_vector.data.uniform_(-stdv, stdv)
    #     self.value_vector.data.uniform_(-stdv, stdv)

    def forward(self, query, value,aux_emb):
        [q,k,v]=torch.split(aux_emb,self.feature_dim,dim=-1)
        query = self.query_linear(query)
        q_vector = self.sigmoid(q)
        q_vector=q_vector.unsqueeze(1)
        query = query * q_vector

        k_vector = self.sigmoid(k)
        k_vector = k_vector.unsqueeze(1)
        key = value * k_vector

        v_vector = self.sigmoid(v)
        v_vector = v_vector.unsqueeze(1)
        if not self.infer:
            v_sigm = self.sigmoid(self.value_linear_sig(v_vector))
            v_tanh = self.tanh(self.value_linear_tan(v_vector))
            v_vector = v_sigm * v_tanh
        value = value * v_vector

        weight = query.matmul(key.transpose(1, 2)) / math.sqrt(self.feature_dim)
        # TODO  上面的可能会造成weight有inf，non_causal版本里没更新，下面更新了在fast版本里
        # weight = query.matmul(key.transpose(1, 2) / math.sqrt(self.feature_dim))
        if self.causal:
            # torch.ones_like会继承目标的梯度，是否有影响？
            mask = torch.tril(torch.ones_like(weight))
            weight = weight * mask
            weight[weight == 0.] = float('-inf')
        weight = self.softmax(weight)
        # TODO  上面的经过softmax可以会造成上溢或下溢从而导致nan，non_casual版本里未更新，下面更新了在fast版本
        # weight = self.softmax(weight - torch.max(weight, dim=-1, keepdim=True)[0])

        out = weight.matmul(value)
        return out

class feedforward_block(nn.Module):
    def __init__(self, feature_dim, ):
        super(feedforward_block, self).__init__()
        self.feature_dim = feature_dim
        self.input_proj = nn.Linear(self.feature_dim, 4 * self.feature_dim)
        self.gussian_elu = nn.GELU()
        self.dropout = nn.Dropout(0.1)

    def forward(self, input):
        out = self.input_proj(input)
        out = self.gussian_elu(out)
        out = self.dropout(out)
        out = torch.split(out, self.feature_dim, -1)
        return out[0] + out[1] + out[2] + out[3]

class arn_block(nn.Module):
    def __init__(self, feature_dim, infer=False, causal=True):
        super(arn_block, self).__init__()
        self.feature_dim = feature_dim
        self.infer = infer
        self.causal = causal
        self.input_norm = nn.LayerNorm(self.feature_dim)
        self.rnn = nn.LSTM(input_size=self.feature_dim,
                           hidden_size=self.feature_dim if causal else (self.feature_dim // 2), num_layers=1,
                           batch_first=True,
                           bidirectional=(not self.causal))
        self.value_norm = nn.LayerNorm(self.feature_dim)
        self.query_norm = nn.LayerNorm(self.feature_dim)
        self.attention = attn_block(self.feature_dim, self.infer, self.causal)
        self.feed_norm = nn.LayerNorm(self.feature_dim)
        self.out_norm = nn.LayerNorm(self.feature_dim)
        self.feedforward = feedforward_block(self.feature_dim)

    def forward(self, input,aux_emb):
        self.rnn.flatten_parameters()
        out = self.input_norm(input)
        out, _ = self.rnn(out)
        value = self.value_norm(out)
        query = self.query_norm(out)
        out = self.attention(query, value,aux_emb)
        out = query + out
        feed_in = self.feed_norm(out)
        out = self.out_norm(out)
        feed_out = self.feedforward(feed_in)
        out = out + feed_out

        return out

class attn_block_anchor(nn.Module):

    def __init__(self, feature_dim, infer=False, causal=True):
        super(attn_block_anchor, self).__init__()
        self.feature_dim = feature_dim
        self.infer = infer
        self.causal = causal
        self.query_vector = nn.Parameter(torch.FloatTensor(self.feature_dim, ), requires_grad=True)
        self.key_vector = nn.Parameter(torch.FloatTensor(self.feature_dim, ), requires_grad=True)
        self.value_vector = nn.Parameter(torch.FloatTensor(self.feature_dim), requires_grad=True)
        self.query_linear = nn.Linear(self.feature_dim, self.feature_dim)
        self.value_linear_sig = nn.Linear(self.feature_dim, self.feature_dim)
        self.value_linear_tan = nn.Linear(self.feature_dim, self.feature_dim)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=-1)
        self.reset_parameters()

    def reset_parameters(self):
    #     # torch.nn.init.xavier_normal_(self.query_vector)
    #     # torch.nn.init.xavier_normal_(self.key_vector)
    #     # torch.nn.init.xavier_normal_(self.value_vector)
        stdv = 1. / math.sqrt(self.query_vector.size(0))
        self.query_vector.data.uniform_(-stdv, stdv)
        self.key_vector.data.uniform_(-stdv, stdv)
        self.value_vector.data.uniform_(-stdv, stdv)

    def forward(self, query, value):
        query = self.query_linear(query)
        q_vector = self.sigmoid(self.query_vector)
        # q_vector=q_vector.unsqueeze(1)
        query = query * q_vector

        k_vector = self.sigmoid(self.key_vector)
        # k_vector = k_vector.unsqueeze(1)
        key = value * k_vector

        v_vector = self.sigmoid(self.value_vector)
        # v_vector = v_vector.unsqueeze(1)
        if not self.infer:
            v_sigm = self.sigmoid(self.value_linear_sig(v_vector))
            v_tanh = self.tanh(self.value_linear_tan(v_vector))
            v_vector = v_sigm * v_tanh
        value = value * v_vector

        weight = query.matmul(key.transpose(1, 2)) / math.sqrt(self.feature_dim)
        # TODO  上面的可能会造成weight有inf，non_causal版本里没更新，下面更新了在fast版本里
        # weight = query.matmul(key.transpose(1, 2) / math.sqrt(self.feature_dim))
        if self.causal:
            # torch.ones_like会继承目标的梯度，是否有影响？
            mask = torch.tril(torch.ones_like(weight))
            weight = weight * mask
            weight[weight == 0.] = float('-inf')
        weight = self.softmax(weight)
        # TODO  上面的经过softmax可以会造成上溢或下溢从而导致nan，non_casual版本里未更新，下面更新了在fast版本
        # weight = self.softmax(weight - torch.max(weight, dim=-1, keepdim=True)[0])

        out = weight.matmul(value)
        return out

class arn_block_anchor(nn.Module):
    def __init__(self, feature_dim, infer=False, causal=True):
        super(arn_block_anchor, self).__init__()
        self.feature_dim = feature_dim
        self.infer = infer
        self.causal = causal
        self.input_norm = nn.LayerNorm(self.feature_dim)
        self.rnn = nn.LSTM(input_size=self.feature_dim,
                           hidden_size=self.feature_dim if causal else (self.feature_dim // 2), num_layers=1,
                           batch_first=True,
                           bidirectional=(not self.causal))
        self.value_norm = nn.LayerNorm(self.feature_dim)
        self.query_norm = nn.LayerNorm(self.feature_dim)
        self.attention = attn_block_anchor(self.feature_dim, self.infer, self.causal)
        self.feed_norm = nn.LayerNorm(self.feature_dim)
        self.out_norm = nn.LayerNorm(self.feature_dim)
        self.feedforward = feedforward_block(self.feature_dim)

    def forward(self, input):
        self.rnn.flatten_parameters()
        norm_out = self.input_norm(input)
        value = self.value_norm(norm_out)
        query = self.query_norm(norm_out)
        att_out = self.attention(query, value)
        att_out = query + att_out
        packed_out=torch.nn.utils.rnn.pack_padded_sequence(att_out,batch_first=True,enforce_sorted=False)
        rnn_out, _ = self.rnn(packed_out)
        pad_out=torch.nn.utils.rnn.pad_packed_sequence(rnn_out,batch_first=True)
        feed_in = self.feed_norm(pad_out)
        out = self.out_norm(pad_out)
        feed_out = self.feedforward(feed_in)
        out = out + feed_out

        return out

class attn_block_separate(nn.Module):

    def __init__(self, feature_dim, infer=False, causal=True):
        super(attn_block_separate, self).__init__()
        self.feature_dim = feature_dim
        self.infer = infer
        self.causal = causal
        self.query_vector=nn.Sequential(
            nn.Linear(in_features=feature_dim,out_features=feature_dim,bias=True),
            nn.PReLU(),
            nn.LayerNorm(feature_dim)
                                        )
        self.key_vector = nn.Sequential(
            nn.Linear(in_features=feature_dim, out_features=feature_dim, bias=True),
            nn.PReLU(),
            nn.LayerNorm(feature_dim)
        )
        self.value_vector = nn.Sequential(
            nn.Linear(in_features=feature_dim, out_features=feature_dim, bias=True),
            nn.PReLU(),
            nn.LayerNorm(feature_dim)
        )
        # self.query_vector = nn.Parameter(torch.FloatTensor(self.feature_dim, ), requires_grad=True)
        # self.key_vector = nn.Parameter(torch.FloatTensor(self.feature_dim, ), requires_grad=True)
        # self.value_vector = nn.Parameter(torch.FloatTensor(self.feature_dim), requires_grad=True)
        self.query_linear = nn.Linear(self.feature_dim, self.feature_dim)
        self.value_linear_sig = nn.Linear(self.feature_dim, self.feature_dim)
        self.value_linear_tan = nn.Linear(self.feature_dim, self.feature_dim)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, query, value, aux_emb,input_len):
        # todo: 要加入非线性
        query = self.query_linear(query)
        q_vector = self.sigmoid(self.query_vector(aux_emb))
        query = query * q_vector


        k_vector = self.sigmoid(self.key_vector(aux_emb))
        key = value * k_vector

        v_vector = self.sigmoid(self.value_vector(aux_emb))
        if not self.infer:
            v_sigm = self.sigmoid(self.value_linear_sig(v_vector))
            v_tanh = self.tanh(self.value_linear_tan(v_vector))
            v_vector = v_sigm * v_tanh
        value = value * v_vector

        weight = query.matmul(key.transpose(1, 2)) / math.sqrt(self.feature_dim)
        # TODO  上面的可能会造成weight有inf，non_causal版本里没更新，下面更新了在fast版本里
        # weight = query.matmul(key.transpose(1, 2) / math.sqrt(self.feature_dim))
        if self.causal:
            # torch.ones_like会继承目标的梯度，是否有影响？
            mask = torch.tril(torch.ones_like(weight))
            weight = weight * mask
            weight[weight == 0.] = float('-inf')
        batch,t,t=weight.shape
        masked_weight_lst=[]
        for i in range(batch):
            tmp = weight[i]
            mask = torch.ones(input_len[i], input_len[i])
            pad_row = torch.zeros(input_len[i], t - input_len[i])
            pad_col = torch.ones(t - input_len[i], t)
            # rank=torch.distributed.get_rank()
            # tmp_mask = torch.cat((torch.cat((mask, pad_row), dim=1), pad_col), dim=0)
            device=tmp.get_device()
            tmp_mask = torch.cat((torch.cat((mask, pad_row), dim=1), pad_col), dim=0).to(device)
            tmp = tmp.masked_fill(tmp_mask == 0, float('-inf'))
            one_maks=torch.ones(input_len[i],t)
            zero_mask=torch.zeros(t - input_len[i], t)
            # full_mask = torch.cat((one_maks, zero_mask), dim=0)
            full_mask=torch.cat((one_maks,zero_mask),dim=0).to(device)
            masked_weight_lst.append(tmp*full_mask)
        masked_weight = torch.stack(masked_weight_lst, dim=0)

        # weight = self.softmax(masked_weight)
        final_weight = self.softmax(masked_weight)

        out = final_weight.matmul(value)
        return out

class arn_block_separate(nn.Module):
    def __init__(self, feature_dim, infer=False, causal=True):
        super(arn_block_separate, self).__init__()
        self.feature_dim = feature_dim
        self.infer = infer
        self.causal = causal
        self.input_norm = nn.LayerNorm(self.feature_dim)
        self.rnn = nn.LSTM(input_size=self.feature_dim,
                           hidden_size=self.feature_dim if causal else (self.feature_dim // 2), num_layers=1,
                           batch_first=True,
                           bidirectional=(not self.causal))
        self.value_norm = nn.LayerNorm(self.feature_dim)
        self.query_norm = nn.LayerNorm(self.feature_dim)
        self.attention = attn_block_separate(self.feature_dim, self.infer, self.causal)
        self.feed_norm = nn.LayerNorm(self.feature_dim)
        self.out_norm = nn.LayerNorm(self.feature_dim)
        self.feedforward = feedforward_block(self.feature_dim)

    def forward(self, input,aux_emb,input_len):
        # torch.backends.cudnn.enabled = False
        self.rnn.flatten_parameters()

        value = self.value_norm(input)
        query = self.query_norm(input)
        att_out = self.attention(query, value,aux_emb,input_len)
        att_out = query + att_out
        b,t,f=att_out.shape
        att_out = self.input_norm(att_out)
        packed_out=torch.nn.utils.rnn.pack_padded_sequence(att_out,input_len.cpu(),batch_first=True)
        rnn_out, _ = self.rnn(packed_out)
        pad_out,_=torch.nn.utils.rnn.pad_packed_sequence(rnn_out,batch_first=True,total_length=t)
        feed_in = self.feed_norm(pad_out)
        out = self.out_norm(pad_out)
        feed_out = self.feedforward(feed_in)
        out = out + feed_out

        return out

class attn_block_separate_cat(nn.Module):

    def __init__(self, feature_dim, infer=False, causal=True):
        super(attn_block_separate_cat, self).__init__()
        self.feature_dim = feature_dim
        self.infer = infer
        self.causal = causal
        self.query_vector = nn.Sequential(
            nn.Linear(in_features=feature_dim*2, out_features=feature_dim, bias=True),
            nn.PReLU(),
            nn.LayerNorm(feature_dim)
        )
        self.key_vector = nn.Sequential(
            nn.Linear(in_features=feature_dim*2, out_features=feature_dim, bias=True),
            nn.PReLU(),
            nn.LayerNorm(feature_dim)
        )
        self.value_vector = nn.Sequential(
            nn.Linear(in_features=feature_dim*2, out_features=feature_dim, bias=True),
            nn.PReLU(),
            nn.LayerNorm(feature_dim)
        )
        
        # self.query_vector = nn.Parameter(torch.FloatTensor(self.feature_dim, ), requires_grad=True)
        # self.key_vector = nn.Parameter(torch.FloatTensor(self.feature_dim, ), requires_grad=True)
        # self.value_vector = nn.Parameter(torch.FloatTensor(self.feature_dim), requires_grad=True)
        self.query_linear = nn.Linear(self.feature_dim, self.feature_dim)
        self.value_linear_sig = nn.Linear(self.feature_dim, self.feature_dim)
        self.value_linear_tan = nn.Linear(self.feature_dim, self.feature_dim)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=-1)


    def forward(self, query, value, aux_emb,input_len):


        query = self.query_linear(query)
        query=torch.cat((query, aux_emb.repeat(1, query.size(1), 1)), dim=-1)
        query = self.query_vector(query)


        key = torch.cat((value,aux_emb.repeat(1,query.size(1),1)),dim=-1)
        key = self.key_vector(key)
        

        value = torch.cat((value,aux_emb.repeat(1,query.size(1),1)),dim=-1)
        v_vector = self.value_vector(value)
        if not self.infer:
            v_sigm = self.sigmoid(self.value_linear_sig(v_vector))
            v_tanh = self.tanh(self.value_linear_tan(v_vector))
            v_vector = v_sigm * v_tanh
        
        value=v_vector


        weight = query.matmul(key.transpose(1, 2)) / math.sqrt(self.feature_dim)
        # TODO  上面的可能会造成weight有inf，non_causal版本里没更新，下面更新了在fast版本里
        # weight = query.matmul(key.transpose(1, 2) / math.sqrt(self.feature_dim))
        if self.causal:
            # torch.ones_like会继承目标的梯度，是否有影响？
            mask = torch.tril(torch.ones_like(weight))
            weight = weight * mask
            weight[weight == 0.] = float('-inf')
        batch, t, t = weight.shape
        masked_weight_lst = []
        for i in range(batch):
            tmp = weight[i]
            mask = torch.ones(input_len[i], input_len[i])
            pad_row = torch.zeros(input_len[i], t - input_len[i])
            pad_col = torch.ones(t - input_len[i], t)
            # rank=torch.distributed.get_rank()
            device = tmp.get_device()
            tmp_mask = torch.cat((torch.cat((mask, pad_row), dim=1), pad_col), dim=0).to(device)
            tmp = tmp.masked_fill(tmp_mask == 0, float('-inf'))
            one_maks = torch.ones(input_len[i], t)
            zero_mask = torch.zeros(t - input_len[i], t)
            full_mask = torch.cat((one_maks, zero_mask), dim=0).to(device)
            masked_weight_lst.append(tmp * full_mask)
        masked_weight = torch.stack(masked_weight_lst, dim=0)

        # weight = self.softmax(masked_weight)
        final_weight = self.softmax(masked_weight)

        out = final_weight.matmul(value)
        return out

class arn_block_separate_cat(nn.Module):
    def __init__(self, feature_dim, infer=False, causal=True):
        super(arn_block_separate_cat, self).__init__()
        self.feature_dim = feature_dim
        self.infer = infer
        self.causal = causal
        self.input_norm = nn.LayerNorm(self.feature_dim)
        self.rnn = nn.LSTM(input_size=self.feature_dim,
                           hidden_size=self.feature_dim if causal else (self.feature_dim // 2), num_layers=1,
                           batch_first=True,
                           bidirectional=(not self.causal))
        self.value_norm = nn.LayerNorm(self.feature_dim)
        self.query_norm = nn.LayerNorm(self.feature_dim)
        self.attention = attn_block_separate_cat(self.feature_dim, self.infer, self.causal)
        self.feed_norm = nn.LayerNorm(self.feature_dim)
        self.out_norm = nn.LayerNorm(self.feature_dim)
        self.feedforward = feedforward_block(self.feature_dim)

    def forward(self, input,aux_emb,input_len):
        # torch.backends.cudnn.enabled = False
        self.rnn.flatten_parameters()

        value = self.value_norm(input)
        query = self.query_norm(input)
        att_out = self.attention(query, value,aux_emb,input_len)
        att_out = query + att_out
        b,t,f=att_out.shape
        att_out = self.input_norm(att_out)
        packed_out=torch.nn.utils.rnn.pack_padded_sequence(att_out,input_len.cpu(),batch_first=True)
        rnn_out, _ = self.rnn(packed_out)
        pad_out,_=torch.nn.utils.rnn.pad_packed_sequence(rnn_out,batch_first=True,total_length=t)
        feed_in = self.feed_norm(pad_out)
        out = self.out_norm(pad_out)
        feed_out = self.feedforward(feed_in)
        out = out + feed_out

        return out

class attn_block_separate_add(nn.Module):

    def __init__(self, feature_dim, infer=False, causal=True):
        super(attn_block_separate_add, self).__init__()
        self.feature_dim = feature_dim
        self.infer = infer
        self.causal = causal
        self.query_vector = nn.Sequential(
            nn.Linear(in_features=feature_dim, out_features=feature_dim, bias=True),
            nn.PReLU(),
            nn.LayerNorm(feature_dim)
        )
        self.key_vector = nn.Sequential(
            nn.Linear(in_features=feature_dim, out_features=feature_dim, bias=True),
            nn.PReLU(),
            nn.LayerNorm(feature_dim)
        )
        self.value_vector = nn.Sequential(
            nn.Linear(in_features=feature_dim, out_features=feature_dim, bias=True),
            nn.PReLU(),
            nn.LayerNorm(feature_dim)
        )
        # self.query_vector = nn.Parameter(torch.FloatTensor(self.feature_dim, ), requires_grad=True)
        # self.key_vector = nn.Parameter(torch.FloatTensor(self.feature_dim, ), requires_grad=True)
        # self.value_vector = nn.Parameter(torch.FloatTensor(self.feature_dim), requires_grad=True)
        self.query_linear = nn.Linear(self.feature_dim, self.feature_dim)
        self.value_linear_sig = nn.Linear(self.feature_dim, self.feature_dim)
        self.value_linear_tan = nn.Linear(self.feature_dim, self.feature_dim)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, query, value, aux_emb, input_len):
        # todo: 要加入非线性
        query = self.query_linear(query)
        query = query + aux_emb.repeat(1,query.size(1),1)
        query = self.query_vector(query)
        
        
        key = value + aux_emb.repeat(1,query.size(1),1)
        key = self.key_vector(key)
       
        
        value = value + aux_emb.repeat(1,query.size(1),1)
        v_vector = self.value_vector(value)
        if not self.infer:
            v_sigm = self.sigmoid(self.value_linear_sig(v_vector))
            v_tanh = self.tanh(self.value_linear_tan(v_vector))
            v_vector = v_sigm * v_tanh
        value=v_vector

        weight = query.matmul(key.transpose(1, 2)) / math.sqrt(self.feature_dim)
        # TODO  上面的可能会造成weight有inf，non_causal版本里没更新，下面更新了在fast版本里
        # weight = query.matmul(key.transpose(1, 2) / math.sqrt(self.feature_dim))
        if self.causal:
            # torch.ones_like会继承目标的梯度，是否有影响？
            mask = torch.tril(torch.ones_like(weight))
            weight = weight * mask
            weight[weight == 0.] = float('-inf')
        batch, t, t = weight.shape
        masked_weight_lst = []
        for i in range(batch):
            tmp = weight[i]
            mask = torch.ones(input_len[i], input_len[i])
            pad_row = torch.zeros(input_len[i], t - input_len[i])
            pad_col = torch.ones(t - input_len[i], t)
            # rank=torch.distributed.get_rank()
            device = tmp.get_device()
            tmp_mask = torch.cat((torch.cat((mask, pad_row), dim=1), pad_col), dim=0).to(device)
            tmp = tmp.masked_fill(tmp_mask == 0, float('-inf'))
            one_maks = torch.ones(input_len[i], t)
            zero_mask = torch.zeros(t - input_len[i], t)
            full_mask = torch.cat((one_maks, zero_mask), dim=0).to(device)
            masked_weight_lst.append(tmp * full_mask)
        masked_weight = torch.stack(masked_weight_lst, dim=0)

        # weight = self.softmax(masked_weight)
        final_weight = self.softmax(masked_weight)

        out = final_weight.matmul(value)
        return out

class arn_block_separate_add(nn.Module):
    def __init__(self, feature_dim, infer=False, causal=True):
        super(arn_block_separate_add, self).__init__()
        self.feature_dim = feature_dim
        self.infer = infer
        self.causal = causal
        self.input_norm = nn.LayerNorm(self.feature_dim)
        self.rnn = nn.LSTM(input_size=self.feature_dim,
                           hidden_size=self.feature_dim if causal else (self.feature_dim // 2), num_layers=1,
                           batch_first=True,
                           bidirectional=(not self.causal))
        self.value_norm = nn.LayerNorm(self.feature_dim)
        self.query_norm = nn.LayerNorm(self.feature_dim)
        self.attention = attn_block_separate_add(self.feature_dim, self.infer, self.causal)
        self.feed_norm = nn.LayerNorm(self.feature_dim)
        self.out_norm = nn.LayerNorm(self.feature_dim)
        self.feedforward = feedforward_block(self.feature_dim)

    def forward(self, input,aux_emb,input_len):
        # torch.backends.cudnn.enabled = False
        self.rnn.flatten_parameters()

        value = self.value_norm(input)
        query = self.query_norm(input)
        att_out = self.attention(query, value,aux_emb,input_len)
        att_out = query + att_out
        b,t,f=att_out.shape
        att_out = self.input_norm(att_out)
        packed_out=torch.nn.utils.rnn.pack_padded_sequence(att_out,input_len.cpu(),batch_first=True)
        rnn_out, _ = self.rnn(packed_out)
        pad_out,_=torch.nn.utils.rnn.pad_packed_sequence(rnn_out,batch_first=True,total_length=t)
        feed_in = self.feed_norm(pad_out)
        out = self.out_norm(pad_out)
        feed_out = self.feedforward(feed_in)
        out = out + feed_out

        return out