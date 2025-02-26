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

##########################

ntokens = 255 # the size of vocabulary
ninp = 4*128 # embedding dimension
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

# 假设这些类和函数已经定义好了
# from some_module import TransformerLayer, TransformerModule, PositionalEncoding, DGCNN, get_graph_feature, knn


class EHEM(nn.Module):
    def __init__(self,win_len, ntoken, ninp, nhead, nhid, nlayers, dropout=0.5, args_dgcnn=None):
        super(EHEM, self).__init__()
        self.model_type = 'EHEM'
        self.ninp = ninp  # Transformer特征维度
        self.win_len = win_len
        self.shift_size = win_len // 2
        
        # DGCNN 特征提取器
        self.dgcnn = DGCNN(k=20,emb_dims=ninp, output_channels=ninp)

        # successive localized self attention 
        self.hsa1 = SlidingWindowEncoder(TransformerLayer(ninp, nhead, nhid, dropout),self.shift_size)
        self.hsa2 = SlidingWindowEncoder(TransformerLayer(ninp, nhead, nhid, dropout),self.shift_size)

        # successive localized attention 
        self.hca1 = SlidingWindowEncoder(TransformerLayer(ninp, nhead, nhid, dropout),self.shift_size)
        self.hca2 = SlidingWindowEncoder(TransformerLayer(ninp, nhead, nhid, dropout),self.shift_size//2)
        # 节点合并 (Node Merging)
        self.node_merging1 = nn.Linear(2 * ninp, ninp)

        # MLP 预测头
        self.decoder0 = nn.Linear(win_len*3//2, win_len)
        self.decoder1 = nn.Linear(ninp, ntoken)
        self.trans2 = nn.Linear(bptt,bptt//2)
        self.act = nn.ReLU()

        self.embedding_xi1 = nn.Linear(255,ninp)
        self.trans1 = nn.Linear(bptt,bptt//2)
        self.node_merging2 = nn.Linear(2 * ninp, ninp)
        self.node_merging3 = nn.Linear(2 * ninp, ninp)
        self.decoder2 = nn.Linear(bptt*5//4 ,bptt//2)
        self.decoder3 = nn.Linear(ninp, ntoken)
        
        self.init_weights()

    def init_weights(self):
        self.decoder0.bias.data.zero_()
        self.decoder0.weight.data = nn.init.xavier_normal_(self.decoder0.weight.data)
        self.decoder1.bias.data.zero_()
        self.decoder1.weight.data = nn.init.xavier_normal_(self.decoder1.weight.data)

    def forward(self, src, src_mask = None, dataFeat=None):
       # src: [bptt, batch_size, K, 6]  (K=4)
       bptt, batch_size, K, _ = src.shape

       # DGCNN 需要输入所有节点的祖辈节点的特征
       # [bptt, batch, K, feature_dim]->[batch_size, num_points, 3, 6]
       dgcnn_input = src[:, :, 0:-1, :].permute(1,0,2,3)

       # 将最后两个维度展平为一个维度
       #  -> [batch_size, num_points, num_dims]
       dgcnn_input = dgcnn_input.view(batch_size, bptt, -1)  

       # 2. DGCNN 特征提取
       # [batch_size, num_points, num_dims]->[batch_size*num_points//win_len,num_dims,win_len]
       dgcnn_input = dgcnn_input.reshape(batch_size*bptt//win_len,win_len,18).permute(0,2,1)

        # 送入 DGCNN
        # outputs:[batch_size*num_points//win_len = batch, win_len, 512]
       Fia = self.dgcnn(dgcnn_input)
       # 3. 分层自注意力
       # f1:[batch, win_len, 512]
       f1 = self.hsa1(Fia,Fia)
       # 相邻向量拼接
       # [batch, win_len//2, 1024]
       outputs = f1.reshape(-1,win_len//2,ninp*2)
       # 节点合并
       # [batch, win_len//2, 512]
       outputs = self.node_merging1(outputs)
       # 再次经过分层自注意力
       # outputs:[batch, win_len//2, 512]
       outputs = self.hsa2(outputs,outputs)
       # concate [batch_size*bptt//win_len,win_len*3//2,ninp]
       outputs = torch.cat([f1,outputs],dim=1)
       
       # 经过mlp得到Fia_hat [batch_size*bptt//win_len,win_len,ninp]
       Fia_hat = self.act(self.decoder0(outputs.transpose(1,2))).transpose(1,2)
       # Fia_hat [batch_size,bptt,ninp]
       Fia_hat = Fia_hat.reshape(batch_size,bptt,ninp)
       
       # 每个节点得到各个编码的分数 [batch_size,bptt//2,ntoken]
       output_xi1 = self.decoder1(self.trans2(Fia_hat.transpose(1,2)).transpose(1,2))
    
       # 5. 兄弟特征嵌入 
       # [batch_size, bptt//2, ninp]
       xi1_embed = self.embedding_xi1(output_xi1)
       
       # [batch_size, bptt//2, ninp]
       Fia1_hat = Fia_hat[:,0::2,:] # 第一组的祖先特征
       Fia2_hat = Fia_hat[:,1::2,:] # 第二组的祖先特征
       
       # [batch_size, bptt, ninp]
       Fi1 = torch.cat([xi1_embed,Fia1_hat],dim=1)
       # [batch_size, bptt//2, ninp]
       Fi1 = self.act(self.trans1(Fi1.transpose(1,2))).transpose(1,2)

       # 交叉注意力
       # [batch_size, bptt//2, ninp]
       f2 = self.hca1(Fia2_hat,Fi1)
    
       # 相邻向量拼接
       # ->[batch_size, bptt//4, ninp*2]
       k_v2 = f2.reshape(-1,bptt//4,ninp*2)
       # 节点合并
       # [batch_size, bptt//4, ninp]
       k_v2 = self.node_merging2(k_v2)
       q2 = self.node_merging3(Fia2_hat.reshape(-1,bptt//4,ninp*2))
       # [batch_size, bptt//4, ninp]
       q2 = self.hca2(q2,k_v2)
       
       # 拼接q2 Fi2 f2
       # q2 [batch_size, bptt//4, ninp]
       # f2 [batch_size, bptt//2, ninp]
       # Fia2_hat [batch_size, bptt//2, ninp]
       # output2 [batch_size, bptt*5//4, ninp]
       output2 = torch.cat([q2,f2,Fia2_hat],dim=1)
       
       # 7. 预测第二组 (xi2)
       # output2 [batch_size, bptt//2, ninp]
       output2 = self.decoder2(output2.transpose(1,2)).transpose(1,2)
       # output_xi2 [batch_size,bptt//2,ntoken]
       output_xi2 = self.decoder3(output2)
       
       return torch.cat([output_xi1,output_xi2],dim=1)


######################################################################
# SlidingWindowTransformer
#
import torch
import torch.nn.functional as F

import torch
import torch.nn.functional as F

class SlidingWindowEncoder(torch.nn.Module):
    def __init__(self, encoder_layer, win_size):
        super().__init__()
        self.encoder_layer = encoder_layer  # 单层 Transformer Layer
        self.win_size = win_size  # 窗口大小

    def forward(self, query_em, key_value_em, mask=None):
        batch_size, win_len, feature_dim = query_em.shape
        assert win_len % self.win_size == 0, "win_len 必须能被 win_size 整除"
        
        # 1. 划分窗口
        num_windows = win_len // self.win_size
        query_em = query_em.view(batch_size, num_windows, self.win_size, feature_dim)  # [B, num_win, win_size, D]
        key_value_em = key_value_em.view(batch_size, num_windows, self.win_size, feature_dim)  # [B, num_win, win_size, D]

        # 2. 局部注意力（第一层 Transformer）
        attn_out = self.encoder_layer(query_em, key_value_em, mask)  # 交叉注意力或自注意力
    
        # 3. 窗口移动（Shift Window）
        shift_attn_out = torch.roll(attn_out, shifts=-self.win_size//2, dims=1)  # 右移半个窗口

        # 4. 第二层 Transformer 注意力
        shift_attn_out = self.encoder_layer(shift_attn_out, shift_attn_out, mask)

        # 5. 逆向窗口移动（Reverse Shift）
        restored_attn_out = torch.roll(shift_attn_out, shifts=self.win_size//2, dims=1)

        # 6. 还原形状
        restored_attn_out = restored_attn_out.view(batch_size, win_len, feature_dim)  # [B, win_len, D]

        return restored_attn_out




######################################################################
# DGCNN
#
import torch.nn.functional as F

class DGCNN(nn.Module):
    def __init__(self, k=20,emb_dims=256,dropout=0.5, output_channels=512):
        super(DGCNN, self).__init__()
        self.k = k
        
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm1d(emb_dims)

        self.conv1 = nn.Sequential(nn.Conv2d(36, 64, kernel_size=1, bias=False),
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
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv6 = nn.Sequential(
            nn.Conv1d(emb_dims, 512, kernel_size=1, bias=False),  # 替换 linear1
            nn.BatchNorm1d(512),
            nn.LeakyReLU(negative_slope=0.2)
        )
        

    def forward(self, x):
        # [batch, 18, win_len]
        x = x.float()
        batch_size = x.size(0)
        # [batch, 36, win_len, k]
        x = get_graph_feature(x, k=self.k)
        pass
        
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

        x = self.conv5(x)          # 输出形状: [batch_size, emb_dims, N]
        x = self.conv6(x)          # 输出形状: [batch_size, 512, N]
        x = x.permute(0, 2, 1)     # 调整维度为 [batch_size, N, 512]
        return x
    
######################################################################
# ``knn`` module 
#
def knn(x, k):
    inner = -2*torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
 
    idx = pairwise_distance.topk(k=k, dim=-1)[1]   # (batch_size, num_points, k)
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
    # (batch_size, num_points, k*num_dims)  -> (batch_size*num_points, k*num_dims) 
    #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size*num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims) 
    # [batch, win_len, k, 18]
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)

    # 拼接差异特征和原始特征，得到 36 维增强特征
    # [batch, 36, win_len, k]
    feature = torch.cat((feature-x, x), dim=3).permute(0, 3, 1, 2).contiguous()
  
    return feature

######################################################################
# ``PositionalEncoding`` module 
#

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

######################################################################
# Functions to generate input and target sequence
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#

def get_batch(source, i):
    # i = 0,bptt - 1,2*bptt-1...
    # 取 bptt 和 len(source) - 1 - i 的较小值
    #source [bptt,batch_size,4,6]
    seq_len = min(bptt, len(source) - 1 - i)
    pad_len = bptt - seq_len
    data = source[i:i+seq_len].clone()
    data = torch.cat([
            data,
            torch.zeros((pad_len, *data.shape[1:]),  # 保持其他维度一致
                       dtype=data.dtype,
                       device=data.device)
        ], dim=0)
    # 目标 target 是 source，用于训练预测下一步的状态
    target = source[i:i+seq_len,:,-1,0]
    return data[:,:,-levelNumK:,:], (target).long(),[]



######################################################################
# Run the model
# -------------
#


model = EHEM(win_len ,ntokens, ninp, nhead, nhid, nlayers, dropout).to(device)


if __name__=="__main__":
    import dataset
    import torch.utils.data as data
    import time
    import os

    epochs = 2 # The number of epochs
    best_model = None
    batch_size = 128
    TreePoint = bptt*16
    train_set = dataset.DataFolder(root=trainDataRoot, TreePoint=TreePoint,transform=None,dataLenPerFile= 294625.44) # you should run 'dataLenPerFile' in dataset.py to get this num (17456051.4)
    train_loader = data.DataLoader(dataset=train_set, batch_size=batch_size, shuffle=False, num_workers=4,drop_last=True) # will load TreePoint*batch_size at one time
    
    # loger
    if not os.path.exists(checkpointPath):
        os.makedirs(checkpointPath)
    printl = CPrintl(expName+'/loss.log')
    writer = SummaryWriter('./log/'+expName)
    printl(datetime.datetime.now().strftime('\r\n%Y-%m-%d:%H:%M:%S'))
    # model_structure(model,printl)
    printl(expComment+' Pid: '+str(os.getpid()))
    # 计算日志间隔
    # batch_size代表train_loader每次取数据的批次大小
    # batchSize代表输入transformer的数据批次大小
    log_interval = int(batch_size*TreePoint/batchSize/bptt)
    
    # learning
    criterion = nn.CrossEntropyLoss()
    lr = 1e-3 # learning rate
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)
    best_val_loss = float("inf")
    idloss = 0

    # reload
    saveDic = None
    # saveDic = reload(100030,checkpointPath)
    if saveDic:
        scheduler.last_epoch = saveDic['epoch'] - 1
        idloss = saveDic['idloss']
        best_val_loss = saveDic['best_val_loss']
        model.load_state_dict(saveDic['encoder'])
        
    def train(epoch):
        global idloss,best_val_loss
        model.train() # Turn on the train mode
        total_loss = 0.
        start_time = time.time()
        total_loss_list = torch.zeros((1,7))
            
        for Batch, d in enumerate(train_loader): # there are two 'BATCH', 'Batch' includes batch_size*TreePoint/batchSize/bptt 'batch'es.
            batch = 0
            # Batch 是当前的大批次索引
            # reshape((batchSize,-1,4,6))调整至batchSize=32,,K,6
            #shape [-1,batch_size,4,6]
            train_data = d[0].reshape((batchSize,-1,4,6)).to(device).permute(1,0,2,3)   
            # bptt：截断反向传播长度，每次取 bptt 个时间步的数据
            for index, i in enumerate(range(0, train_data.size(0) - 1, bptt)):
                # data [bptt, batch, K, feature_dim]
                # K:祖先 0->孩子 3
                # targets [bptt,batch]
                data, targets,dataFeat = get_batch(train_data, i)#data [35,20]
                optimizer.zero_grad()
                
                # [batch,bptt,ntokens]
                # targets:[bptt,batch]
                output = model(data, src_mask=None,dataFeat=dataFeat)                         #output: [bptt,batch size,255]
                # print(targets.shape,targets.shape[0],output.shape)
                output = output[:,:targets.shape[0], :]
                
                loss = criterion(output.reshape(-1,ntokens)
                                 ,targets.permute(1,0).reshape(-1))/math.log(2)
                
                loss.backward()
                # 限制梯度范数，防止梯度爆炸
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                # 更新模型参数
                optimizer.step()
                total_loss += loss.item()
                batch = batch+1

                # 每 log_interval 记录一次损失
                if batch % log_interval == 0:
                    cur_loss = total_loss / log_interval
                    elapsed = time.time() - start_time
                
                    total_loss_list = " - "
                    # math.exp(cur_loss) 计算困惑度 (Perplexity, PPL)
                    printl('| epoch {:3d} | Batch {:3d} | {:4d}/{:4d} batches | '
                        'lr {:g} | ms/batch {:5.2f} | '
                        'loss {:5.2f} | losslist  {} | ppl {:8.2f}'.format(
                            epoch, Batch, batch, len(train_data) // bptt, scheduler.get_last_lr()[0],
                            elapsed * 1000 / log_interval,
                            cur_loss,total_loss_list, math.exp(cur_loss)))
                    total_loss = 0
    
                    start_time = time.time()

                    writer.add_scalar('train_loss', cur_loss,idloss)
                    idloss+=1
            #  每 10 个 Batch 保存模型 Exp/Obj/checkpoint
            if Batch%10==0:
                save(epoch*100000+Batch,saveDict={'encoder':model.state_dict(),'idloss':idloss,'epoch':epoch,'best_val_loss':best_val_loss},modelDir=checkpointPath)
    
    # train
    for epoch in range(1, epochs + 1):
        epoch_start_time = time.time()
        train(epoch)
        printl('-' * 89)
        scheduler.step()
        printl('-' * 89)
