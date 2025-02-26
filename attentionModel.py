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

    def __init__(self, emsize, nhead, dropout=0.5):
        super(MultiheadAttention, self).__init__()
        self.nhead = nhead  # 4
        self.head_size = emsize // nhead  
        assert self.head_size * nhead == emsize, "embed_dim must be divisible by num_heads"

        self.all_head_size = int(self.nhead * self.head_size)   
        self.mlpKey = nn.Linear(emsize, self.all_head_size)  
        self.mlpQuery = nn.Linear(emsize, self.all_head_size)
        self.mlpValue = nn.Linear(emsize, self.all_head_size)
        self.dropout = nn.Dropout(dropout)
        
    # Slice the output of mlpKQV to implement multi-head attention.
    def slice(self,x,dim):
        new_x_shape = x.size()[:-1] + (self.nhead, self.head_size)  
        x = x.view(*new_x_shape)
        if (dim == 3):
            x = x.permute(0, 2, 1, 3)
        elif (dim == 4):
            x = x.permute(0,1,3,2,4)
            # assert 0
        return x

    #em.shape = [bptt,batch_size,emsize]  mask.shape=[bptt, bptt]
    def forward(self,query_em, key_value_em,mask):
        query_em = query_em.transpose(0, 1).contiguous()
        key_value_em = key_value_em.transpose(0, 1).contiguous()

        # 生成Q, K, V
        Query = self.mlpQuery(query_em)
        Key = self.mlpKey(key_value_em)
        Value = self.mlpValue(key_value_em)

        # 分割多头
        Query = self.slice(Query, query_em.dim())  # [batch, nhead, q_len, head_size]
        Key = self.slice(Key, key_value_em.dim())  # [batch, nhead, kv_len, head_size]
        Value = self.slice(Value, key_value_em.dim())

        attention_score = torch.matmul(Query, Key.transpose(-1, -2)) / math.sqrt(self.head_size)   
        if mask is not None:
            mask = mask.unsqueeze(0).unsqueeze(0)  # [1, 1, q_len, kv_len]
            attention_scores = attention_scores + mask
        attention_map = self.dropout(nn.Softmax(dim=-1)(attention_score))

        context = torch.matmul(attention_map, Value)        
        if (context.dim() == 4):
            context = context.permute(0, 2, 1, 3).contiguous()  
        elif (context.dim()==5):
            context = context.permute(0, 1, 3, 2, 4).contiguous()   
        context_shape = context.size()[:-2] + (self.all_head_size,)
        context = context.view(*context_shape)
        context = context.transpose(0,1).contiguous()
        return context
 
class TransformerLayer(nn.Module):

    def __init__(self, ninp, nhead, nhid, dropout=0.1):
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
    def forward(self, query, key_value, src_mask = None):
        attn_output = self.MultiAttention(query, key_value,src_mask)  #Multi-head Attention
        query = self.norm1(attn_output)
        ff_output = self.linear2(self.dropout(torch.relu(self.linear1(query))))  
        query = query + self.dropout2(ff_output)
        query = self.norm2(query)
        return query

class TransformerModule(nn.Module):

    def __init__(self,layer, nlayers):
        super(TransformerModule, self).__init__()
        self.layers = torch.nn.ModuleList([copy.deepcopy(layer) for i in range(nlayers)])

    def forward(self,query, key_value,src_mask = None):
        output = query

        for mod in self.layers:
            output = mod(output, key_value, src_mask)
        return output
