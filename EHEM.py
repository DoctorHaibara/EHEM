'''
Author: fuchy@stu.pku.edu.cn
Date: 2021-09-20 08:06:11
LastEditTime: 2021-09-20 23:53:24
LastEditors: fcy
Description: the training file
             see networkTool.py to set up the parameters
             will generate training log file loss.log and checkpoint in folder 'expName'
FilePath: /compression/octAttention.py
All rights reserved.
'''
import math
import torch
import torch.nn as nn
import os
import datetime
from networkTool import *
from torch.utils.tensorboard import SummaryWriter
from attentionModel import TransformerLayer,TransformerModule
from collections import defaultdict
##########################

ntokens = 255 # the size of vocabulary
ninp = 4*(128+4+6) # embedding dimension
win_len = 32 # local window size

nhid = 300 # the dimension of the feedforward network model in nn.TransformerEncoder
nlayers = 3 # the number of nn.TransformerEncoderLayer in nn.TransformerEncoder
nhead = 4 # the number of heads in the multiheadattention models
dropout = 0 # the dropout value
batchSize = 32



import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpyAc

class EHEMA(nn.Module):
    def __init__(self,win_len, ntoken, ninp, nhead, nhid, nlayers, dropout=0, args_dgcnn=None):
        super(EHEMA, self).__init__()
        self.model_type = 'EHEMA'
        self.ninp = ninp  
        self.win_len = win_len
        self.shift_size = win_len // 2

        self.encoder = nn.Embedding(ntoken+1, 128,padding_idx=255)
        self.encoder1 = nn.Embedding(MAX_OCTREE_LEVEL+1, 6)
        self.encoder2 = nn.Embedding(9, 4)

        self.encoder3 = nn.Embedding(ntoken,ninp,padding_idx=0)

        self.dgcnn = DGCNN(k=20,emb_dims=ninp, output_channels=ninp)

        # successive localized self attention 
        self.hsa = HierarchicalAttention(TransformerLayer(ninp, nhead, nhid, dropout),
                                          bptt,bptt,ninp)              # 暂时将窗口大小设置为 bptt，相当于将整个序列划分为一个窗口的特殊情况
                                          # win_len,bptt,ninp)

        self.multiscale_proj1 = DimensionCompressionLayer()
        self.multiscale_proj2 = nn.Conv1d(in_channels=ninp,  out_channels=ninp, kernel_size=2,stride=2)
        self.decoder1 = nn.Linear(ninp, ntoken)

    def forward(self, Fia):
        Fia1 = Fia[:,0::2,:]
        batch_size,seq_len,_ = Fia.shape
    
        output = self.hsa(Fia,Fia)
        Fia_hat = self.multiscale_proj1(seq_len+1,output) 

        output_xi = self.decoder1(self.multiscale_proj2(Fia_hat.permute(0,2,1)).permute(0,2,1) + Fia1)
        return output_xi
        

class EHEMB(nn.Module):
    def __init__(self,win_len, ntoken, ninp, nhead, nhid, nlayers, dropout=0, args_dgcnn=None):
        super(EHEMB, self).__init__()
        self.model_type = 'EHEMB'
        self.ninp = ninp  
        self.win_len = win_len
        self.shift_size = win_len // 2

        self.encoder3 = nn.Embedding(ntoken,ninp)

        # successive localized self attention 
        self.compress1 = nn.Conv1d(in_channels=ninp,  out_channels=ninp, kernel_size=2,stride=2)
        self.compress2 = nn.Conv1d(in_channels=ninp,  out_channels=ninp, kernel_size=4,stride=2)
        self.multiscale_proj2 = DimensionCompressionLayer()

        # successive localized cross attention 
        self.hca = HierarchicalAttention(TransformerLayer(ninp, nhead, nhid, dropout),
                                         bptt//2,bptt//2,ninp)
                                        # win_len,bptt//2,ninp)
        self.multiscale_proj3 = nn.Conv1d(in_channels=ninp,  out_channels=ninp, kernel_size=2,stride=2)
        self.decoder2 = nn.Linear(ninp ,ntoken)

    def forward(self, Fia_hat,xi1):
        Fia1_hat = Fia_hat[:,0::2,:]
        Fia2_hat = Fia_hat[:,1::2,:]
        seq1_len,seq2_len = Fia1_hat.shape[1],Fia2_hat.shape[1]
        
        xi1_embed = self.encoder3(xi1.unsqueeze(0))
        if seq1_len != seq2_len:
            Fi1 = self.compress2(torch.cat([xi1_embed,Fia1_hat],dim=1).permute(0,2,1)).permute(0,2,1)
        else:
            Fi1 = self.compress1(torch.cat([xi1_embed,Fia1_hat],dim=1).permute(0,2,1)).permute(0,2,1)
        assert Fi1.shape[1] == seq2_len

        output = self.hca(Fia2_hat,Fi1)
        output = self.multiscale_proj2(seq2_len,output) + Fia2_hat

        output_xi2 = self.decoder2(output)
        return output_xi2



import torch.nn.functional as F
class DimensionCompressionLayer(nn.Module):
    def __init__(self, kernel_size=2, stride=2,feature_dim =ninp):
        super(DimensionCompressionLayer, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.feature_dim = feature_dim
        self.conv1d = nn.Conv1d(in_channels=ninp, out_channels=ninp, 
                                kernel_size=self.kernel_size, stride=self.stride) # 卷积层在 __init__ 中创建，参数固定

    def forward(self, target_seqlen, x):
        input_length = x.shape[1] # 从输入张量中获取 input_length
        padding_length = 2 * target_seqlen - input_length
        # 创建 padding 层，padding 大小在 forward 中动态计算
        padded_x = F.pad(x, (0,0,0,max(0, padding_length)), mode='constant', value=0) # padding_length 可能为负数，用 max(0, padding_length) 避免负 padding

        # 应用卷积
        output = self.conv1d(padded_x.permute(0,2,1)).permute(0,2,1)
        return output


def generate_masks(batch_size, seq_len, padded_seq_len, padding_len, win_size, offset, device):
    
    # 生成 padding_mask
    if padding_len > 0:
        padding_mask = torch.zeros((batch_size, padded_seq_len, padded_seq_len), dtype=torch.bool, device=device)
        padding_mask[:, :, seq_len+offset:seq_len+padding_len+offset] = True  # 填充部分为 True (表示需要掩码)
        padding_mask[:, seq_len+offset:seq_len+padding_len+offset, :] = True
    else:
        padding_mask = torch.zeros((batch_size, padded_seq_len, padded_seq_len), dtype=torch.bool, device=device)

    # 计算窗口数量
    num_windows = padded_seq_len // win_size

    # 将 padding_mask 转换为窗口化掩码
    windowed_mask = padding_mask.view(batch_size, num_windows, win_size, num_windows, win_size).permute(0, 1, 3, 2, 4)
    # [win_num,1,win_size,win_size]
    windowed_mask = torch.diagonal(windowed_mask, dim1=1, dim2=2).permute( 3,0, 1, 2)

    return windowed_mask


######################################################################
# SlidingWindowTransformer
#
class SlidingWindowEncoder(torch.nn.Module):
    def __init__(self, encoder_layer, win_size):
        super().__init__()
        self.encoder_layer = encoder_layer  # 单层 Transformer Layer
        self.win_size = win_size  # 窗口大小

    def forward(self, query_em, key_value_em):
        batch_size, seq_len, feature_dim = query_em.shape

        padding_len = 0
        if seq_len % self.win_size != 0:
            padding_len = self.win_size - (seq_len % self.win_size)
        
        if padding_len > 0:
            query_em = F.pad(query_em, (0, 0, 0, padding_len)) # 在序列长度维度 (dim=1) 的末尾 padding

            key_value_em = F.pad(key_value_em, (0, 0, 0, padding_len)) # 同样 padding key_value_em
            padded_seq_len = seq_len + padding_len
        else:
            padded_seq_len = seq_len

        num_windows = padded_seq_len // self.win_size
        
        window_padding_mask = generate_masks(batch_size, seq_len, padded_seq_len, padding_len, self.win_size, offset=0, device=query_em.device)
    
        query_em = query_em.view(batch_size*num_windows, self.win_size, feature_dim)  # [B*num_win, win_size, D]
        key_value_em = key_value_em.view(batch_size*num_windows, self.win_size, feature_dim)  # [B*num_win, win_size, D]
        
        attn_out = self.encoder_layer(query_em, key_value_em, key_padding_mask=window_padding_mask) 
        attn_out = attn_out.view(batch_size,-1,feature_dim)
        
        if padding_len > 0:
            attn_out = attn_out[:, :seq_len, :] #  移除 padding 部分
        return attn_out 
     
        shift_attn_out = torch.roll(attn_out, shifts=(-self.win_size//2), dims=1).view(batch_size, num_windows, self.win_size, feature_dim)
        window_padding_mask = generate_masks(batch_size, seq_len, padded_seq_len, padding_len, 
                                             self.win_size, offset=-self.win_size//2, device=query_em.device)
    
        shift_attn_out = self.encoder_layer(shift_attn_out, shift_attn_out, key_padding_mask=window_padding_mask) # 假设 encoder_layer 接受 key_padding_mask
        shift_attn_out = shift_attn_out.view(batch_size,-1,feature_dim)

        restored_attn_out = torch.roll(shift_attn_out, shifts=(self.win_size//2), dims=1)

        restored_attn_out = restored_attn_out.view(batch_size, num_windows * self.win_size, feature_dim) # 恢复形状先
        if padding_len > 0:
            restored_attn_out = restored_attn_out[:, :seq_len, :] #  移除 padding 部分

        return restored_attn_out

######################################################################
# SlidingWindowTransformer
#

class HierarchicalAttention(nn.Module):
    def __init__(self, encoder_layer, win_size, seq_len, feature_dim):
        super().__init__()
        self.encoder_layer = encoder_layer
        self.win_size = win_size
        self.seq_len = seq_len
        self.feature_dim = feature_dim

        # 最大层数
        self.max_num_layers = int(math.log2(seq_len // win_size)) + 1

        # 每层的注意力模块
        self.attention_layers = nn.ModuleList([
            SlidingWindowEncoder(encoder_layer, win_size)
            for i in range(self.max_num_layers)
        ])

        # 下采样层
        self.downsample_q = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(feature_dim, feature_dim, kernel_size=2, stride=2),
                nn.GELU()
            ) for _ in range(self.max_num_layers - 1)
        ])
        self.downsample_kv = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(feature_dim, feature_dim, kernel_size=2, stride=2),
                nn.GELU()
            ) for _ in range(self.max_num_layers - 1)
        ])

    def forward(self, query_em, key_value_em):
        batch_size, seq_len, feature_dim = query_em.shape

        all_outputs = []
        current_q = query_em
        current_kv = key_value_em  # 初始化独立的key_value输入
        layer_num =  int(math.ceil(math.log2(math.ceil(seq_len/ self.win_size)))) + 1

        assert layer_num == 1 # 对应当前只划分一个窗口的情况

        for i in range(self.max_num_layers - layer_num,self.max_num_layers):
            # 1. 应用当前层的滑动窗口注意力
            layer_output = self.attention_layers[i](current_q, current_kv)
            all_outputs.append(layer_output)  # 保存当前层输出

            # 2. 下采样（除了最后一层）
            if i < self.max_num_layers - 1:
                # 下采样query：使用当前层的输出
                current_q = self.downsample_q[i](current_q.transpose(1, 2)).transpose(1, 2)
                # 下采样key_value：使用当前的key_value输入
                current_kv = self.downsample_kv[i](current_kv.transpose(1, 2)).transpose(1, 2)

        # 3. 沿序列维度拼接所有层输出
        final_output = torch.cat(all_outputs, dim=1)
        assert len(final_output) == 1
        return final_output

######################################################################
# DGCNN
#
import torch.nn.functional as F

class DGCNN(nn.Module):
    def __init__(self, k=10,emb_dims=256,dropout=0.5, output_channels=512):
        super(DGCNN, self).__init__()
        self.k = k
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm1d(emb_dims)

        self.conv1 = nn.Sequential(nn.Conv2d(48, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64*2, 128, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(128*2, 256, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv1d(512, emb_dims, kernel_size=1, bias=False),
                                   #self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv6 = nn.Sequential(
            nn.Conv1d(emb_dims, ninp, kernel_size=1, bias=False),  # 替换 linear1
            nn.LeakyReLU(negative_slope=0.2)
        )
        self.decoder1 = nn.Linear(ninp,ninp)
        self.decoder2 = nn.Linear(ninp,ntokens)

    def forward(self, x):
        seq_len,batch_size, K, _ = x.shape

        dgcnn_input = x.permute(1,0,2,3)

        x = dgcnn_input.reshape(batch_size, seq_len, -1).permute(0,2,1)
        x = x.float()
        x = get_graph_feature(x, k=self.k)
        
        x = self.conv1(x)
        x1 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x1, k=self.k)
        x = self.conv2(x)
        x2 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x2, k=self.k)
        x = self.conv3(x)
        x3 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x3, k=self.k)
        x = self.conv4(x)
        x4 = x.max(dim=-1, keepdim=False)[0]

        x = torch.cat((x1, x2, x3, x4), dim=1)
        x = self.conv5(x)          # [batch_size, emb_dims, N]
        x = self.conv6(x)          # [batch_size, 512, N]
        feature = x.permute(0, 2, 1)     # [batch_size, N, 512]
        x = self.decoder2(self.decoder1(feature))
        return feature,x
    
######################################################################
# ``knn`` module 
#
def knn(x, k):
    inner = -2*torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
    
    num_points = x.size(2)
    k_effective = min(k, num_points)
    idx = pairwise_distance.topk(k=k_effective, dim=-1)[1]   # (batch_size, num_points, k)
    
    if k_effective < k: 
        padding_indices = idx[:, :, -1].unsqueeze(-1).repeat(1, 1, k - k_effective) 
        idx = torch.cat([idx, padding_indices], dim=-1) 
    return idx


######################################################################
# ``get_graph_feature`` module 
#

def get_graph_feature(x, k=20, idx=None):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx = knn(x, k=k)   # (batch_size, num_points, k)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 每个元素是0, num_points, 2num_points,...
    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1)*num_points

    idx = idx + idx_base

    # idx 展平
    idx = idx.view(-1)
 
    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()   
    
    feature = x.view(batch_size*num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims) 
    # [batch, win_len, k, 18]
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)

    # [batch, 36, win_len, k]
    feature = torch.cat((feature-x, x), dim=3).permute(0, 3, 1, 2).contiguous()
  
    return feature


######################################################################
# Functions to generate input and target sequence
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#

def get_batch(source, level):
    mask = source[:,:,-1,1] == level
    mask = mask.squeeze()

    data = source[mask,:,:,:]

    data[:,:,:-1,0]  # [0-254]
    data[:,:,-1,0] = 255 # pad
    
    target = source[mask,:,-1,0].squeeze(-1)
    
    return data,(target).long(),[]


def network_creator(layer_list, in_channels):
    layers = []
    in_channels = in_channels
    for v in layer_list:
        mlp_layer = nn.Linear(in_channels, v)
        layers += [mlp_layer, nn.ReLU()]
        in_channels = v
    return layers


class OctSqueezeNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.feature_layers = nn.ModuleList(network_creator([128, 128, 128, 128, 128], 6))
        self.aggregation_layers1 = nn.ModuleList(network_creator([128, 128, 128], 128 * 2))
        self.aggregation_layers2 = nn.ModuleList(network_creator([128, 128, 128], 128 * 2))
        self.softmax = nn.Softmax(dim=1)
        self.last_linear = nn.Linear(256, 255)

    def forward(self, data):
        data = data.float()
        cur_node = data[:,:,3,:]
        parent_1 = data[:,:,2,:]
        parent_2 = data[:,:,1,:]
        parent_3 = data[:,:,0,:]
        for layer in self.feature_layers:
            cur_node = layer(cur_node)
        for layer in self.feature_layers:
            parent_1 = layer(parent_1)
        for layer in self.feature_layers:
            parent_2 = layer(parent_2)
        for layer in self.feature_layers:
            parent_3 = layer(parent_3)
        aggregation_c_p1 = torch.cat((cur_node, parent_1), dim=-1)
        aggregation_c_p1 = self.aggregation_layers1[0](aggregation_c_p1)
        aggregation_c_p1 = self.aggregation_layers1[1](aggregation_c_p1)
        for k in range(2, len(self.aggregation_layers1), 2):
            aggregation_c_p1 = aggregation_c_p1 + self.aggregation_layers1[k](aggregation_c_p1)
            aggregation_c_p1 = self.aggregation_layers1[k + 1](aggregation_c_p1)

        aggregation_p1_p2 = torch.cat((parent_1, parent_2), dim=-1)
        aggregation_p1_p2 = self.aggregation_layers1[0](aggregation_p1_p2)
        aggregation_p1_p2 = self.aggregation_layers1[1](aggregation_p1_p2)
        for k in range(2, len(self.aggregation_layers1), 2):
            aggregation_p1_p2 = aggregation_p1_p2 + self.aggregation_layers1[k](aggregation_p1_p2)
            aggregation_p1_p2 = self.aggregation_layers1[k + 1](aggregation_p1_p2)

        aggregation_c_p1_p2 = torch.cat((aggregation_c_p1, aggregation_p1_p2), dim=-1)
        aggregation_c_p1_p2 = self.aggregation_layers2[0](aggregation_c_p1_p2)
        aggregation_c_p1_p2 = self.aggregation_layers2[1](aggregation_c_p1_p2)
        for k in range(2, len(self.aggregation_layers2), 2):
            aggregation_c_p1_p2 = aggregation_c_p1_p2 + self.aggregation_layers2[k](aggregation_c_p1_p2)
            aggregation_c_p1_p2 = self.aggregation_layers2[k + 1](aggregation_c_p1_p2)

        aggregation_p2_p3 = torch.cat((parent_2, parent_3), dim=-1)
        aggregation_p2_p3 = self.aggregation_layers1[0](aggregation_p2_p3)
        aggregation_p2_p3 = self.aggregation_layers1[1](aggregation_p2_p3)
        for k in range(2, len(self.aggregation_layers1), 2):
            aggregation_p2_p3 = aggregation_p2_p3 + self.aggregation_layers1[k](aggregation_p2_p3)
            aggregation_p2_p3 = self.aggregation_layers1[k + 1](aggregation_p2_p3)

        aggregation_p1_p2_p3 = torch.cat((aggregation_p1_p2, aggregation_p2_p3), dim=-1)
        aggregation_p1_p2_p3 = self.aggregation_layers2[0](aggregation_p1_p2_p3)
        aggregation_p1_p2_p3 = self.aggregation_layers2[1](aggregation_p1_p2_p3)
        for k in range(2, len(self.aggregation_layers2), 2):
            aggregation_p1_p2_p3 = aggregation_p1_p2_p3 + self.aggregation_layers2[k](aggregation_p1_p2_p3)
            aggregation_p1_p2_p3 = self.aggregation_layers2[k + 1](aggregation_p1_p2_p3)

        aggregation_c_p1_p2_p3 = torch.cat((aggregation_c_p1_p2, aggregation_p1_p2_p3), dim=-1)

        feature = aggregation_c_p1_p2_p3.squeeze()
        out = self.last_linear(feature)

        return out


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        output = output.squeeze(0)
        _, pred = output.topk(maxk, -1, True, True)

        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


######################################################################
# Run the model
# -------------
#

# EHEM
shared_backbone = DGCNN(k=20,emb_dims=ninp, output_channels=ninp).to(device)
module_a = EHEMA(win_len ,ntokens, ninp, nhead, nhid, nlayers, dropout).to(device)
module_b = EHEMB(win_len ,ntokens, ninp, nhead, nhid, nlayers, dropout).to(device)

# OctSqueezeNet
# model = OctSqueezeNet().to(device) 


if __name__=="__main__":
    import dataset
    import torch.utils.data as data
    import time
    import os

    epochs = 32 # The number of epochs
    best_model = None
    batch_size = 1
    TreePoint = bptt*16
    train_set = dataset.DataFolder(root=trainDataRoot, TreePoint=TreePoint,transform=None,dataLenPerFile= 358181.855) # you should run 'dataLenPerFile' in dataset.py to get this num (17456051.4)
    train_loader = data.DataLoader(dataset=train_set, batch_size=batch_size, shuffle=False, num_workers=4,drop_last=True) # will load TreePoint*batch_size at one time
    
    # loger
    if not os.path.exists(checkpointPath):
        os.makedirs(checkpointPath)
    printl = CPrintl(expName+'/loss.log')
    writer = SummaryWriter('./log/'+expName)
    printl(datetime.datetime.now().strftime('\r\n%Y-%m-%d:%H:%M:%S'))
    printl(expComment+' Pid: '+str(os.getpid()))
    # 计算日志间隔
    log_interval = 64
    
    # learning
    criterion = nn.CrossEntropyLoss(label_smoothing=0.15)
    lr = 1e-3 # learning rate
    optimizer = torch.optim.Adam([
                            {'params': shared_backbone.parameters()},
                            {'params': module_a.parameters()},
                            {'params': module_b.parameters()}
                        ], lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)
    best_val_loss = float("inf")
    idloss = 0

    # reload
    saveDic = None
    # saveDic = reload(100030,checkpointPath)
    # if saveDic:
    #     scheduler.last_epoch = saveDic['epoch'] - 1
    #     idloss = saveDic['idloss']
    #     best_val_loss = saveDic['best_val_loss']
    #     model.load_state_dict(saveDic['encoder'])
        
    def train(epoch):
        global idloss,best_val_loss
        shared_backbone.train() # Turn on the train mode
        module_a.train()
        module_b.train()
        total_loss = total_loss_a = total_loss_b = 0.
        start_time = time.time()
        total_loss_list = torch.zeros((1,7))
        loss_list = []
        batch = 0
        list_acc1_a = list_acc1_b = []
        list_acc5_a = list_acc5_b = []
        for Batch, d in enumerate(train_loader): # there are two 'BATCH', 'Batch' includes batch_size*TreePoint/batchSize/bptt 'batch'es.
            train_data = d[0].reshape((-1,batch_size,4,6)).to(device)

            levels = np.unique(train_data[:,:,-1,1].cpu().numpy())
            for level in levels:
                data, targets, dataFeat = get_batch(train_data, level+2)#data [35,20]
                optimizer.zero_grad()
                for left in range(0,data.shape[0],bptt):
                    if left + bptt >= data.shape[0]:
                        right = data.shape[0]
                    else:
                        right = left + bptt
                    input = data[left:right,:,:,:]
                    target = targets[left:right]  

                    xi1,xi2 = target[0::2],target[1::2] 

                    features,output = shared_backbone(input)
                    features_stopped = features.detach()
                    loss = criterion(output.reshape(-1,ntokens)
                                     ,target)/math.log(2)
                    

                    output_xi1 = module_a(features_stopped)

                    loss_a = criterion(output_xi1.reshape(-1,ntokens)
                                     ,xi1)/math.log(2)
                    acc1_a, acc5_a = accuracy(output_xi1, xi1, topk=(1, 5))
                    list_acc1_a.append(acc1_a)
                    list_acc5_a.append(acc5_a)
                    
                    if xi2.shape[0] > 0:
                        output_xi2 = module_b(features_stopped,xi1)
                        loss_b = criterion(output_xi2.reshape(-1,ntokens)
                                        ,xi2)/math.log(2)
                        
                        acc1_b, acc5_b = accuracy(output_xi2, xi2, topk=(1, 5))
                        list_acc1_b.append(acc1_b)
                        list_acc5_b.append(acc5_b)
                        loss_b.backward()
                        total_loss_b += loss_b.item()

                    loss.backward()
                    loss_a.backward()
                    
                    # 限制梯度范数，防止梯度爆炸
                    torch.nn.utils.clip_grad_norm_(shared_backbone.parameters(), 0.5)
                    torch.nn.utils.clip_grad_norm_(module_a.parameters(), 0.5)
                    torch.nn.utils.clip_grad_norm_(module_b.parameters(), 0.5)
                    # 更新模型参数
                    optimizer.step()
                    total_loss += loss.item()
                    total_loss_a += loss_a.item()
                    
                    batch = batch+1
                    if batch % log_interval == 0:
                        
                        cur_loss = total_loss / log_interval
                        cur_loss_a = total_loss_a / log_interval
                        cur_loss_b = total_loss_b / log_interval
                        elapsed = time.time() - start_time
                    
                        total_loss_list = " - "
                        printl('| epoch {:3d} | Batch {:3d} | {:4d}/{:4d} batches | '
                            'lr {:g} | ms/batch {:5.2f} | '
                            'loss {:5.2f} | losslist  {} | ppl {:8.2f}'.format(
                                epoch, Batch, batch, log_interval, scheduler.get_last_lr()[0],
                                elapsed * 1000 / log_interval,
                                cur_loss,total_loss_list, math.exp(cur_loss)))
                        total_loss = total_loss_a = total_loss_b = 0
                        print(f"Level {level} - Loss_a: {cur_loss_a}, Acc1_a: {sum(list_acc1_a) / len(list_acc1_a)}, Acc5_a: {sum(list_acc5_a) / len(list_acc5_a)}")
                        print(f"Level {level} - Loss_b: {cur_loss_b}, Acc1_b: {sum(list_acc1_b) / len(list_acc1_b)}, Acc5_b: {sum(list_acc5_b) / len(list_acc5_b)}")
                        list_acc1_a = list_acc5_a = list_acc1_b = list_acc5_b = []
                        start_time = time.time()
                        batch = 0
                        writer.add_scalar('train_loss', cur_loss,idloss)
                        idloss+=1
            
            
        save(epoch*100000+Batch,saveDict={'DGCNN':shared_backbone.state_dict(),'module_a':module_a.state_dict(),'module_b':module_b.state_dict(),
                                            'idloss':idloss,'epoch':epoch,'best_val_loss':best_val_loss},modelDir=checkpointPath)
    
    # train
    for epoch in range(1, epochs + 1):
        epoch_start_time = time.time()
        train(epoch)
        printl('-' * 89)
        scheduler.step()
        printl('-' * 89)
