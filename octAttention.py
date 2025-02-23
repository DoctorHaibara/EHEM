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
ninp = 4*(128+4+6) # embedding dimension


nhid = 300 # the dimension of the feedforward network model in nn.TransformerEncoder
nlayers = 3 # the number of nn.TransformerEncoderLayer in nn.TransformerEncoder
nhead = 4 # the number of heads in the multiheadattention models
dropout = 0 # the dropout value
batchSize = 32



class TransformerModel(nn.Module):

    def __init__(self, ntoken, ninp, nhead, nhid, nlayers, dropout=0.5):
        super(TransformerModel, self).__init__()
        self.model_type = 'Transformer'

        self.pos_encoder = PositionalEncoding(ninp, dropout)

        encoder_layers = TransformerLayer(ninp, nhead, nhid, dropout)
        self.transformer_encoder = TransformerModule(encoder_layers, nlayers)

        self.encoder = nn.Embedding(ntoken, 128)
        self.encoder1 = nn.Embedding(MAX_OCTREE_LEVEL+1, 6)
        self.encoder2 = nn.Embedding(9, 4)

        self.ninp = ninp
        self.act = nn.ReLU()
        self.decoder0 = nn.Linear(ninp, ninp)
        self.decoder1 = nn.Linear(ninp, ntoken)
        self.init_weights()

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data = nn.init.xavier_normal_(self.encoder.weight.data )
        self.decoder0.bias.data.zero_()
        self.decoder0.weight.data= nn.init.xavier_normal_(self.decoder0.weight.data )
        self.decoder1.bias.data.zero_()
        self.decoder1.weight.data = nn.init.xavier_normal_(self.decoder1.weight.data )
    def forward(self, src, src_mask, dataFeat):
        # input 形状：(bptt, 1, K, 6)
        bptt = src.shape[0]
        batch = src.shape[1]

        # 祖先的占用代码（1-255）-1
        oct = src[:,:,:,0] #oct[bptt,batchsize,FeatDim(levels)] [0~254]
        # 节点 i 的祖先 j 的层级编号
        level = src[:,:,:,1]  # [0~12] 0 for padding data
        # 该祖先在其父节点中的卦限编号
        octant = src[:,:,:,2] # [0~8] 0 for padding data

        # assert oct.min()>=0 and oct.max()<255
        # assert level.min()>=0 and level.max()<=12
        # assert octant.min()>=0 and octant.max()<=8
        
        # 对 level 张量进行原地裁剪，使其值在 [0, MAX_OCTREE_LEVEL] 范围内
        level -= torch.clip(level[:,:,-1:] - 10,0,None)# the max level in traning dataset is 10        
        torch.clip_(level,0,MAX_OCTREE_LEVEL) 
        
        # #a[bptt,batchsize,FeatDim(levels),EmbeddingDim]
        aOct = self.encoder(oct.long()) # bptt,batch_size,K,128
        aLevel = self.encoder1(level.long()) # bptt,batch_size,K,6
        aOctant = self.encoder2(octant.long()) # bptt,batch_size,K,4

        # bptt,batch_size,K,138
        a = torch.cat((aOct,aLevel,aOctant),3)
        # 最后一维被展平（flatten）为一个连续的向量
        # # bptt,batch_size,K*138=ninp K==4时，为552
        a = a.reshape((bptt,batch,-1)) 
        
        # src = self.ancestor_attention(a)
        # ninp = 4*(128+4+6) # embedding dimension
        # [bptt, batch_size, FeatDim(levels) * embedding_dim]->[bptt, batch_size, ninp]
        src = a.reshape((bptt,a.shape[1],self.ninp))* math.sqrt(self.ninp)

        # src = self.pos_encoder(src)
        # 经过一个多层的 Transformer 编码器，由多个 TransformerLayer 组成\
        # [bptt, batch_size, ninp]
        output = self.transformer_encoder(src, src_mask)
        # 通过两层线性变换和激活函数
        # [bptt, batch_size, ntoken]
        output = self.decoder1(self.act(self.decoder0(output)))
        return output

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
    # 取 bptt 和 len(source) - 1 - i 的较小值
    seq_len = min(bptt, len(source) - 1 - i)
    data = source[i:i+seq_len].clone()
    # 目标 target 是 时间步向后移动一位 的 source，用于训练预测下一步的状态
    target = source[i+1:i+1+seq_len,:,-1,0].reshape(-1)
    # this moves the feat(octant,level) of current node to lastrow,   
    # 整个矩阵的 0:-1 列沿着第一维节点 (N) 进行了 时间序列上的平移，后面的数据覆盖了前面的数据，类似于队列的滚动更新。      
    data[:,:,0:-1,:] = source[i+1:i+seq_len+1,:,0:-1,:] 
    # which will be used as known feat
    # 确保了 oct_seq 在 最后一列 的 1:3 维度（八叉树层级、象限编号等）也遵循时间序列滚动。
    data[:,:,-1,1:3] = source[i+1:i+seq_len+1,:,-1,1:3]
    return data[:,:,-levelNumK:,:], (target).long(),[]



######################################################################
# Run the model
# -------------
#
model = TransformerModel(ntokens, ninp, nhead, nhid, nlayers, dropout).to(device)
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
            # print(Batch,d)
            batch = 0
            # Batch 是当前的大批次索引
            # reshape((batchSize,-1,4,6))调整至batchSize=32,,K,6
            train_data = d[0].reshape((batchSize,-1,4,6)).to(device).permute(1,0,2,3)   #shape [TreePoint*batch_size(data)/batch_size,batch_size,7,6]
            src_mask = model.generate_square_subsequent_mask(bptt).to(device)
            # bptt：截断反向传播长度，每次取 bptt 个时间步的数据
            for index, i in enumerate(range(0, train_data.size(0) - 1, bptt)):
                data, targets,dataFeat = get_batch(train_data, i)#data [35,20]
                optimizer.zero_grad()
                if data.size(0) != bptt:
                    src_mask = model.generate_square_subsequent_mask(data.size(0)).to(device)
                output = model(data, src_mask,dataFeat)                         #output: [bptt,batch size,255]
                loss = criterion(output.view(-1, ntokens), targets)/math.log(2)
                
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
            #  每 10 个 Batch 保存模型
            if Batch%10==0:
                save(epoch*100000+Batch,saveDict={'encoder':model.state_dict(),'idloss':idloss,'epoch':epoch,'best_val_loss':best_val_loss},modelDir=checkpointPath)
    
    # train
    for epoch in range(1, epochs + 1):
        epoch_start_time = time.time()
        train(epoch)
        printl('-' * 89)
        scheduler.step()
        printl('-' * 89)
