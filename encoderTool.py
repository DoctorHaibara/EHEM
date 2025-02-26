'''
Author: fuchy@stu.pku.edu.cn
Description: The encoder helper
FilePath: /compression/encoderTool.py
'''

#%%
import numpy as np 
import torch
import time
import os
from networkTool import device,bptt,expName,levelNumK,MAX_OCTREE_LEVEL
from dataset import default_loader as matloader
import numpyAc
import tqdm

 
'''
description: Rearrange data for batch processing
'''
def batchify(oct_seq,bptt,oct_len):
    """
    batchify 主要用于对 oct_seq 进行时间序列上的平移处理，
    同时添加填充数据，并返回一个 dataID 以及扩展维度的 oct_seq。
    """
    pad_len = int(np.ceil(len(oct_seq)/bptt)*bptt - len(oct_seq))

    dataID = torch.LongTensor(np.arange(oct_len))
    
    # oct_seq 尾部添加 pad_len 行的 0，保证 oct_seq 长度对 bptt 取整
    # 最终 oct_seq 变成了 (data_len + pad_len, K, 6)，即扩展了 pad_len 维度用于填充
    oct_seq = torch.Tensor(np.r_[oct_seq,np.zeros((pad_len,*oct_seq.shape[1:]))])
    
    # dataID 变成了一个 ((data_len + pad_len),) 的 LongTensor，其中：
    # N 个索引 0, 1, ..., N-1 对应 oct_seq 原始数据部分。
    # 末尾 pad_len 个 -1 代表填充部分。
    dataID = torch.LongTensor(np.r_[np.arange(oct_len),np.ones((pad_len))*-1])
    
    # dataID 变成 (data_len + pad_len, 1)，用于标记时间步的索引。
    # oct_seq 变成 (data_len + pad_len, 1, K, 6)，增加 1 维用于批次处理。
    return dataID.unsqueeze(1),oct_seq.unsqueeze(1)

def encodeNode(pro,octvalue):
    assert octvalue<=255 and octvalue>=1
    # 根据概率分布 pro 计算预测值 pre
    pre = np.argmax(pro)+1
    # 编码当前节点所需的比特数 防止概率为 0 时导致数值错误
    # 预测准确性 相等则返回 1
    return -np.log2(pro[octvalue-1]+1e-07),int(octvalue==pre)
'''
description: compress function
param {N[n treepoints]*K[k ancestors]*C[oct code,level,octant,position(xyz)] array; Octree data sequence} oct_data_seq
param {str;bin file name} outputfile
param {model} model
param {bool;Determines whether to perform entropy coding} actualcode
return {float;estimated/true bin size (in bit)} binsz
return {int;oct length of all octree} oct_len
return {float;total foward time of the model} elapsed
return {float list;estimated bin size (in bit) of depth 8~maxlevel data} binszList
return {int list;oct length of 8~maxlevel octree} octNumList
'''
def compress(oct_data_seq,outputfile,model,actualcode = True,print=print,showRelut=False):
    model.eval()
    # oct_data_seq[:,:,1]为包含节点 i 的祖先 j 的层级编号（从 1 开始）
    # oct_data_seq[:,-1,1]为节点i的level
    levelID = oct_data_seq[:,-1,1].copy()
    oct_data_seq = oct_data_seq.copy()

    if levelID.max()>MAX_OCTREE_LEVEL:
        print('**warning!!**,to clip the level>{:d}!'.format(MAX_OCTREE_LEVEL))
        
    oct_seq = oct_data_seq[:,-1:,0].astype(int)   
    oct_len = len(oct_seq)
 
    batch_size =1  # 1 for safety encoder

    # bptt为 Context window length
    assert(batch_size*bptt<oct_len)
    
    #%%
    dataID,padingdata = batchify(oct_data_seq,bptt,oct_len)
    MAX_GPU_MEM_It = 2**13 # you can change this according to the GPU memory size (2**12 for 24G)
    MAX_GPU_MEM = min(bptt*MAX_GPU_MEM_It,dataID.max())+2  #  bptt <= MAX_GPU_MEM -1 < min(MAX_GPU,dataID)
    # 初始化概率张量，形状 (MAX_GPU_MEM, 255)，存储模型输出的 softmax 结果
    pro = torch.zeros((MAX_GPU_MEM,255)).to(device)
    padingLength = padingdata.shape[0]
    padingdata = padingdata
    # 存储累计计算时间
    elapsed = 0
    # 存储每个批次的 softmax 结果
    proBit = []
    # 于计算 nodeID 位置偏移
    offset = 0
    if not showRelut:
        trange = range
    else:
        trange = tqdm.trange
    with torch.no_grad():
        for n,i in enumerate(trange(0, padingLength , bptt)):
            # input 形状：(bptt, 1, K, 6)
            input = padingdata[i:i+bptt].long().to(device)   #input torch.Size([256, 32, 4, 3]) bptt,batch_sz,kparent,[oct,level,octant]
            nodeID = dataID[i:i+bptt].squeeze(0) - offset
            # nodeID < 0 的部分设为 -1，用于处理填充数据
            nodeID[nodeID<0] = -1
            start_time = time.time()
            output = model(input,dataFeat=[])
            elapsed =elapsed+ time.time() - start_time
            output = output.reshape(-1,255)
            nodeID = nodeID.reshape(-1)
            # 对每行进行 Softmax 计算
            p  = torch.softmax(output,1)
            pro[nodeID,:] = p
            # 存储一次 softmax 结果到 proBit，防止 pro 占用过多 GPU 内存
            if( (n % MAX_GPU_MEM_It==0 and n>0) or n == padingLength//bptt-1):
                proBit.append(pro[:nodeID.max()+1].detach().cpu().numpy())
                offset = offset + nodeID.max() +1

    del pro,input
    torch.cuda.empty_cache()
    # 将 proBit 中的多个数组按行方向堆叠成一个新的二维数组
    proBit = np.vstack(proBit)
    #%%
 
    bit = 0
    acc = 0
    templevel = 1
    binszList = []
    octNumList = []
    if True:
        # Estimate the bitrate at each level
        for i in range(oct_len):
            octvalue = int(oct_seq[i,-1])
            bit0,acc0 =encodeNode(proBit[i],octvalue)
            bit+=bit0
            acc+=acc0
            if templevel!=levelID[i]:
                templevel = levelID[i]
                binszList.append(bit)
                octNumList.append(i+1)
        binszList.append(bit)
        octNumList.append(i+1)
        binsz = bit # estimated bin size

        if actualcode:
            # 使用算术编码器对节点序列进行实际编码
            codec = numpyAc.arithmeticCoding()
            if not os.path.exists(os.path.dirname(outputfile)):
                os.makedirs(os.path.dirname(outputfile))
            _,binsz = codec.encode(proBit[:oct_len,:], oct_seq.astype(np.int16).squeeze(-1)-1,outputfile)
       
        if len(binszList)<=7:
            return binsz,oct_len,elapsed,np.array(binszList),np.array(octNumList)  
        # np.array(binszList)每个层级的比特率 np.array(octNumList)每个层级的节点数量。
        return binsz,oct_len,elapsed ,np.array(binszList[7:]),np.array(octNumList[7:])  
 # %%
def main(fileName,model,actualcode = True,showRelut=True,printl = print):
    
    matDataPath = fileName
    octDataPath = matDataPath
    cell,mat = matloader(matDataPath)
    FeatDim = levelNumK  

    # (N, K, 6) 矩阵：祖先的占用代码（0-255）+ 节点 i 的祖先 j 的层级编号 + 祖先在其父节点中的卦限编号（1-8）+ 每个祖先的3D坐标           
    oct_data_seq = np.transpose(mat[cell[0,0]]).astype(int)[:,-FeatDim:,0:6] 

    # p 为原始点云
    p = np.transpose(mat[cell[1,0]]['Location'])
    ptNum = p.shape[0]
    ptName = os.path.basename(matDataPath)
    outputfile = expName+"/data/"+ptName[:-4]+".bin"
    binsz,oct_len,elapsed,binszList,octNumList = compress(oct_data_seq,outputfile,model,actualcode,printl,showRelut)
    if showRelut:
        printl("ptName: ",ptName)
        printl("time(s):",elapsed)
        # 打印原始数据文件的路径
        printl("ori file",octDataPath)
        printl("ptNum:",ptNum)
        printl("binsize(b):",binsz)
        # Bits Per Input Point
        printl("bpip:",binsz/ptNum)

        np.set_printoptions(formatter={'float': '{: 0.2f}'.format})
        printl("pre sz(b) from Q8:",(binszList))
        # 打印每层中每个八叉树节点的平均比特数
        printl("pre bit per oct from Q8:",(binszList/octNumList))
        printl('octNum',octNumList)
        # 输出每个八叉树节点的比特数
        printl("bit per oct:",binsz/oct_len)
        printl("oct len",oct_len)
 
    return binsz/oct_len
