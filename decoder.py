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
from tqdm import trange
from Octree import DeOctree, dec2bin
import pt 
from dataset import default_loader as matloader
from collections import deque
import os 
import time
from networkTool import *
from encoder import shared_backbone,module_a,module_b,list_orifile
import numpyAc
from EHEM import ntokens
batch_size = 1 
bpttRepeatTime = 1
#%%

def concate(list1, list2):
    """
    将两个列表交替合并。
    """
    output_list = []
    len1, len2 = len(list1), len(list2)
    max_len = max(len1, len2)

    for i in range(max_len):
        if i < len1:
            output_list.append(list1[i])
        if i < len2:
            output_list.append(list2[i])

    return output_list

def entropy_model(KfatherNode,dec,shared_backbone,module_a,module_b):
    """
    模拟熵模型的输出，返回预测的概率分布。
    """
    shared_backbone.eval()
    module_a.eval()
    module_b.eval()

    with torch.no_grad():
        xi = []
        data = torch.tensor(KfatherNode).to(device)
        for left in range(0,data.shape[0],bptt):
                if left + bptt >= data.shape[0]:
                    right = data.shape[0]
                else:
                    right = left + bptt
                input = data[left:right,:,:,:]
                length2 = input.shape[0]//2
                features , _ = shared_backbone(input)
                
                if length2 == 0:
                    output_xi1 = module_a(features)
                    xi.append(decodeNode(output_xi1,dec))
                else:
                    output_xi1 = module_a(features)
                    xi1 = torch.tensor(decodeNode(output_xi1,dec)).to(device) 
                    output_xi2 = module_b(features,xi1-1)
                    xi2 = decodeNode(output_xi2,dec)
                    xi.append(concate(xi1.cpu().detach().numpy().tolist(),xi2))
                    
        return np.concatenate(xi, axis=0)

def decodeNode(pro_batch,dec):
    pro_batch = torch.softmax(pro_batch.squeeze(0),-1).cpu().detach().numpy()
    num_nodes = pro_batch.shape[0]
    roots = []
    for i in range(num_nodes):
        pro = pro_batch[i, :]  # 获取单个节点的概率分布
        root = dec.decode(np.expand_dims(pro, 0)) # 仍然调用单符号解码
        roots.append([root + 1])
    return np.concatenate(roots)


def decodeOct(binfile,oct_data_seq,shared_backbone,module_a,module_b,Lmax=11):
    # 初始化根节点
    KfatherNode = [[254, 0, 0, 0, 0, 0]] * 3 + [[255, 1, 1, 0, 0, 0]]
    current_layer_nodes = [[KfatherNode]]
    parent_coords = [[np.array([0, 0, 0])]]  # 根节点坐标
    oct_len = len(oct_data_seq)
    dec = numpyAc.arithmeticDeCoding(None,oct_len,255,binfile)
    nodeID = 0

    elapsed = time.time()
    for layer_num in range(2,Lmax):
        print(f"Current level:{layer_num}")

        # 通过熵模型预测layer_num - 1的占用 [num_node,255]
        # 并且构建layer_num的输入
        # current layer nodes 为num nodes个 4*6的列表组成
        outputs = entropy_model(current_layer_nodes,dec,shared_backbone,module_a,module_b)
        next_layer_nodes = []
        tmp_coords = []
        
        assert outputs[0] == oct_data_seq[nodeID]
        nodeID += len(outputs)

        for i in trange(len(outputs)):  
            output = outputs[i]          
            # 通过 DeOctree 获取父节点所有的子节点的八叉象限和坐标
            child_coords, child_octants,num_nodes = DeOctree([output], parent_coords[i], layer_num, Lmax)

            # current layer nodes 为num child nodes个 4*6的列表组成
            current_layer_node = current_layer_nodes[i][0]
            # 更新父节点的占用情况
            current_layer_node[3][0] = output - 1  

            for i in range(num_nodes):
                next_layer_nodes.append([current_layer_node[1:]+ [[255, layer_num, child_octants[i], *child_coords[i].tolist()[0]]]])
            
            tmp_coords.append(child_coords)

        tmp_coords = np.concatenate(tmp_coords, axis=0)
        # 更新子节点的特征信息和坐标信息 作为下一层的输入
        current_layer_nodes = next_layer_nodes
        parent_coords = tmp_coords
    return parent_coords.squeeze(1),time.time() - elapsed

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
        ptrec,elapsed = decodeOct(binfile,oct_data_seq,shared_backbone,module_a,module_b)
        print('decode succee,time:', elapsed)
        # Dequantization
        DQpt = (ptrec*qs+offset)
        pt.write_ply_data(expName+"/temp/test/rec.ply",DQpt)
        pt.pcerror(p,DQpt,None,'-r 1',None).wait()
