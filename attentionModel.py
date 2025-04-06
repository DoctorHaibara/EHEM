'''
Author: fuchy@stu.pku.edu.cn
Date: 2021-09-17 23:30:48
LastEditTime: 2021-12-02 22:18:56
LastEditors: FCY SR
Description: attentionModel
FilePath: /compression/attentionModel.py
'''
import torch
import torch.nn as nn
import math
import copy
from networkTool import device

class MultiheadAttention(nn.Module):

    def __init__(self, emsize, nhead, dropout=0):
        super(MultiheadAttention, self).__init__()
        self.nhead = nhead
        self.head_size = emsize // nhead
        assert self.head_size * nhead == emsize, "embed_dim must be divisible by num_heads"

        self.all_head_size = int(self.nhead * self.head_size)
        self.mlpKey = nn.Linear(emsize, self.all_head_size)
        self.mlpQuery = nn.Linear(emsize, self.all_head_size)
        self.mlpValue = nn.Linear(emsize, self.all_head_size)
        self.dropout = nn.Dropout(dropout)

    def slice(self, x, dim):
        new_x_shape = x.size()[:-1] + (self.nhead, self.head_size)
        x = x.view(*new_x_shape)
        if dim == 3:
            x = x.permute(0, 2, 1, 3)
        elif dim == 4:
            x = x.permute(0, 1, 3, 2, 4)
        return x

    def forward(self, query_em, key_value_em, key_padding_mask=None):
        Query = self.mlpQuery(query_em)
        Key = self.mlpKey(key_value_em)
        Value = self.mlpValue(key_value_em)

        Query = self.slice(Query, query_em.dim())
        Key = self.slice(Key, key_value_em.dim())
        Value = self.slice(Value, key_value_em.dim())

        attention_score = torch.matmul(Query, Key.transpose(-1, -2)) / math.sqrt(self.head_size)
        if key_padding_mask is not None:
            attention_score = attention_score.masked_fill(key_padding_mask, -1e9)

        attention_map = nn.Softmax(dim=-1)(attention_score)
        if key_padding_mask is not None:
            attention_map = attention_map.masked_fill(key_padding_mask, 0)

        context = torch.matmul(attention_map, Value)
        if context.dim() == 4:
            context = context.permute(0, 2, 1, 3).contiguous()
        elif context.dim() == 5:
            context = context.permute(0, 1, 3, 2, 4).contiguous()

        context_shape = context.size()[:-2] + (self.all_head_size,)
        context = context.view(*context_shape)
        return context
    
 
class TransformerLayer(nn.Module):

    def __init__(self, ninp, nhead, nhid, dropout=0):
        super(TransformerLayer, self).__init__()
        self.MultiAttention = MultiheadAttention(emsize=ninp,nhead=nhead)
        self.linear1 = nn.Linear(ninp,nhid)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(nhid,ninp)

        self.norm1 = nn.LayerNorm(ninp, eps=1e-5) # It will affect parallel coding 
        self.norm2 = nn.LayerNorm(ninp, eps=1e-5)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    # src is the integration of leaf node and its ancestors.
    def forward(self, query, key_value, key_padding_mask = None):
        attn_output = self.MultiAttention(query, key_value,key_padding_mask)  #Multi-head Attention
        query = self.norm1(attn_output)
        ff_output = self.linear2(torch.relu(self.linear1(query)))
        query = query + ff_output
        query = self.norm2(query)
        return query


class TransformerModule(nn.Module):

    def __init__(self,layer, nlayers):
        super(TransformerModule, self).__init__()
        self.layers = torch.nn.ModuleList([copy.deepcopy(layer) for i in range(nlayers)])

    def forward(self,query, key_value,key_padding_mask = None):
        output = query

        for mod in self.layers:
            output = mod(output, key_value, key_padding_mask)
        return output
