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
from collections import defaultdict
 

def get_batch(source, level):
    mask = source[:,:,-1,1] == level
    mask = mask.squeeze()
    
    data = source[mask,:,:,:]

    assert np.all(data[:,:,:-1,0] <= 254) and np.all(data[:,:,:-1,0] >= 0)
    data[:,:,-1,0] = 255 # pad

    assert np.all(data[:,:,-1,0] == 255)
    target = source[mask,:,-1,0].squeeze(-1)
    
    return torch.from_numpy(data).to(device),(torch.from_numpy(target).to(device)).long()


def encodeNode(pro,octvalue):
    assert octvalue<=255 and octvalue>=1
    # pro 的索引范围 [0,254]
    pre = np.argmax(pro)+1
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
def compress(oct_data_seq,outputfile,shared_backbone,module_a,module_b,actualcode = True,print=print,showRelut=False):
    shared_backbone.eval()
    module_a.eval()
    module_b.eval()
    
    levelID = oct_data_seq[:,-1,1].copy()
    oct_seq = []
    oct_data_seq = oct_data_seq.copy()
    oct_data_seq[:,:,0] -= 1
    
    oct_len = oct_data_seq.shape[0]
    oct_data_seq = np.expand_dims(oct_data_seq, axis=1)
    batch_size =1  # 1 for safety encoder

    assert(batch_size*bptt<oct_len)
    # 存储累计计算时间
    elapsed = 0
    # 存储每个批次的 softmax 结果
    proBit = []

    if not showRelut:
        trange = range
    else:
        trange = tqdm.trange
    with torch.no_grad():
        levels = np.unique(levelID)
        for level in levels:
            print(f"Current level:{level}")
            data, targets = get_batch(oct_data_seq, level)#data [35,20]
            for left in trange(0,data.shape[0],bptt):
                if left + bptt >= data.shape[0]:
                    right = data.shape[0]
                else:
                    right = left + bptt
                input = data[left:right,:,:,:]
                target = targets[left:right]  
                start_time = time.time()
                xi1,xi2 = target[0::2],target[1::2] 
                oct_seq.append(xi1.cpu().numpy()+1)
                oct_seq.append(xi2.cpu().numpy()+1)
                features , _ = shared_backbone(input)              

                output_xi1 = module_a(features)

                if xi2.shape[0] > 0:
                    output_xi2 = module_b(features,xi1)

                elapsed =elapsed + time.time() - start_time
                
                if xi2.shape[0] > 0:
                    output = torch.cat((output_xi1,output_xi2),dim=1)
                else:
                    output = output_xi1
        
                # 对每行进行 Softmax 计算
                p  = torch.softmax(output,-1)
                
                proBit.append(p.detach().cpu().numpy())


    del input
    torch.cuda.empty_cache()

    proBit = np.concatenate(proBit, axis=1)
    proBit = np.squeeze(proBit,axis=0)
    oct_seq = np.concatenate(oct_seq, axis=0)
    assert proBit.shape[0] == oct_len and oct_seq.shape[0] == oct_len
 
    bit = 0
    acc = 0
    binszList = []
    octNumList = []
    if True:
        # Estimate the bitrate at each level
        for i in range(oct_len):
            octvalue = int(oct_seq[i])
            bit0,acc0 =encodeNode(proBit[i],octvalue)
            bit+=bit0
            acc+=acc0
        binszList.append(bit)
        octNumList.append(i+1)
        binsz = bit # estimated bin size

        if actualcode:
            # 使用算术编码器对节点序列进行实际编码
            codec = numpyAc.arithmeticCoding()
            if not os.path.exists(os.path.dirname(outputfile)):
                os.makedirs(os.path.dirname(outputfile))
            _,binsz = codec.encode(proBit[:oct_len,:], oct_seq.astype(np.int16)-1,outputfile)
       
        if len(binszList)<=7:
            return binsz,oct_len,elapsed,np.array(binszList),np.array(octNumList)  
        # np.array(binszList)每个层级的比特率 np.array(octNumList)每个层级的节点数量。
        return binsz,oct_len,elapsed ,np.array(binszList[7:]),np.array(octNumList[7:])  
 # %%
def main(fileName,shared_backbone,module_a,module_b,actualcode = True,showRelut=True,printl = print):
    
    matDataPath = fileName
    octDataPath = matDataPath
    cell,mat = matloader(matDataPath)
    FeatDim = levelNumK  
         
    oct_data_seq = np.transpose(mat[cell[0,0]]).astype(int)[:,-FeatDim:,0:6] 

    # p 为原始点云
    p = np.transpose(mat[cell[1,0]]['Location'])
    ptNum = p.shape[0]
    ptName = os.path.basename(matDataPath)
    outputfile = expName+"/data/"+ptName[:-4]+".bin"
    binsz,oct_len,elapsed,binszList,octNumList = compress(oct_data_seq,outputfile,shared_backbone,module_a,module_b,actualcode,printl,showRelut)
    if showRelut:
        printl("ptName: ",ptName)
        printl("time(s):",elapsed)
        printl("ori file",octDataPath)
        printl("ptNum:",ptNum)
        printl("binsize(b):",binsz)
        # Bits Per Input Point
        printl("bpip:",binsz/ptNum)

        np.set_printoptions(formatter={'float': '{: 0.2f}'.format})
        printl("pre sz(b) from Q8:",(binszList))
        printl("pre bit per oct from Q8:",(binszList/octNumList))
        printl('octNum',octNumList)
        # 输出每个八叉树节点的比特数
        printl("bit per oct:",binsz/oct_len)
        printl("oct len",oct_len)
 
    return binsz/oct_len
