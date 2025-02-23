'''
Author: fuchy@stu.pku.edu.cn
Date: 2021-09-17 23:30:48
LastEditTime: 2021-12-02 22:18:56
LastEditors: FCY
Description: decoder
FilePath: /compression/decoder.py
All rights reserved.
'''
#%%
import  numpy as np
import torch
from tqdm import tqdm
from Octree import DeOctree, dec2bin
import pt 
from dataset import default_loader as matloader
from collections import deque
import os 
import time
from networkTool import *
from encoderTool import generate_square_subsequent_mask
from encoder import model,list_orifile
import numpyAc
batch_size = 1 
bpttRepeatTime = 1
#%%
'''
description: decode bin file to occupancy code
param {str;input bin file name} binfile
param {N*1 array; occupancy code, only used for check} oct_data_seq
param {model} model
param {int; Context window length} bptt
return {N*1,float}occupancy code,time
'''
def decodeOct(binfile,oct_data_seq,model,bptt):
    """
    从二进制文件中解码八叉树的占用码（occupancy code）
    oct_data_seq 为ground truth
    oct_seq 为解码结果
    """

    model.eval()
    with torch.no_grad():
        elapsed = time.time()

        KfatherNode = [[255,0,0]]*levelNumK # # 初始化 K 级父节点
        nodeQ = deque()
        oct_seq = [] # 存储解码后的 occupancy code
        src_mask = generate_square_subsequent_mask(bptt).to(device)

        input = torch.zeros((bptt,batch_size,levelNumK,3)).long().to(device)
        padinginbptt = torch.zeros((bptt,batch_size,levelNumK,3)).long().to(device)
        bpttMovSize = bptt//bpttRepeatTime
        # input torch.Size([256, 32, 4, 3]) bptt,batch_sz,kparent,[oct,level,octant]
        # all of [oct,level,octant] default is zero

        output = model(input,src_mask,[])

        # 初始化解码
        freqsinit = torch.softmax(output[-1],1).squeeze().cpu().detach().numpy()
        
        oct_len = len(oct_data_seq)

        # 创建算术解码器，最大值 255 oct_len表示总共需要解码多少个 occupancy code
        dec = numpyAc.arithmeticDeCoding(None,oct_len,255,binfile)

        root =  decodeNode(freqsinit,dec)
        nodeId = 0
        
        KfatherNode = KfatherNode[3:]+[[root,1,1]] + [[root,1,1]] # for padding for first row # ( the parent of root node is root itself)
        
        nodeQ.append(KfatherNode) 
        oct_seq.append(root) #decode the root  
        
        with tqdm(total=  oct_len+10) as pbar:
            while True:
                father = nodeQ.popleft()
                # 解析当前父节点的占用码（0-255）为二进制列表（8位，表示8个子节点的占用情况）
                childOcu = dec2bin(father[-1][0]) 
                # 使其按照 子节点顺序（0-7） 进行遍历
                childOcu.reverse()
                # 获取当前父节点的层级（深度）
                faterLevel = father[-1][1] 
                for i in range(8):
                    if(childOcu[i]):
                        # root 是 当前子节点编号
                        # faterLevel+1 表示它的层级，子节点层级比父节点高一级
                        # i+1 表示 它是父节点的第几个子节点（1-8）
                        faterFeat = [[father+[[root,faterLevel+1,i+1]]]] 
                        # Fill in the information of the node currently decoded [xi-1, xi level, xi octant]
                        faterFeatTensor = torch.Tensor(faterFeat).long().to(device)
                        # # root 需要 -1 处理，使其变为 0-indexed（原始 root 在 decodeNode 里加了 1）
                        faterFeatTensor[:,:,:,0] -= 1

                        # shift bptt window
                        offsetInbpttt = (nodeId)%(bpttMovSize) # the offset of current node in the bppt window
                        # model 看到的是 bpttMovSize 个 node
                        if offsetInbpttt==0: # a new bptt window
                            input = torch.vstack((
                                input[bpttMovSize:],  # 滑动窗口：丢弃 bpttMovSize 个旧数据
                                faterFeatTensor,  # 添加当前解码的 faterFeatTensor
                                padinginbptt[0:bpttMovSize-1]  # 添加 bpttMovSize-1 个 padding 数据
                            ))
                        else:
                            # 添加当前解码的 faterFeatTensor
                            input[bptt-bpttMovSize+offsetInbpttt] = faterFeatTensor

                        # 在一个bpttMovSize中，input中node的数量从1->bpttMovSize
                        output = model(input,src_mask,[])
                        
                        Pro = torch.softmax(output[offsetInbpttt+bptt-bpttMovSize],1).squeeze().cpu().detach().numpy()

                        root =  decodeNode(Pro,dec)
                        nodeId += 1
                        pbar.update(1)
                        # 构造新的子节点信息
                        KfatherNode = father[1:]+[[root,faterLevel+1,i+1]]
                        nodeQ.append(KfatherNode)
                        if(root==256 or nodeId==oct_len):
                            assert len(oct_data_seq) == nodeId # for check oct num
                            # 记录最终的八叉树序列
                            Code = oct_seq
                            return Code,time.time() - elapsed
                        oct_seq.append(root)
                    assert oct_data_seq[nodeId] == root # for check

def decodeNode(pro,dec):
    # dec.decode() 负责从 pro 的概率分布中解码出一个整数 root，表示当前八叉树节点的占用码（occupancy code）
    root = dec.decode(np.expand_dims(pro,0))
    return root+1


if __name__=="__main__":

    for oriFile in list_orifile: # from encoder.py
        ptName = os.path.basename(oriFile)[:-4]
        matName = 'Data/testPly/'+ptName+'.mat'
        binfile = expName+'/data/'+ptName+'.bin'
        cell,mat =matloader(matName)

        # Read Sideinfo
        oct_data_seq = np.transpose(mat[cell[0,0]]).astype(int)[:,-1:,0]# for check
        
        p = np.transpose(mat[cell[1,0]]['Location']) # ori point cloud
        offset = np.transpose(mat[cell[2,0]]['offset'])
        qs = mat[cell[2,0]]['qs'][0]

        # (N, 1) 矩阵：祖先的占用代码（0-255）
        Code,elapsed = decodeOct(binfile,oct_data_seq,model,bptt)
        print('decode succee,time:', elapsed)
        print('oct len:',len(Code))

        # DeOctree
        ptrec = DeOctree(Code)
        # Dequantization
        DQpt = (ptrec*qs+offset)
        pt.write_ply_data(expName+"/temp/test/rec.ply",DQpt)
        pt.pcerror(p,DQpt,None,'-r 1',None).wait()
