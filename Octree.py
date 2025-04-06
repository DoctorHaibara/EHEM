'''
Author: fuchy@stu.pku.edu.cn
Description: Octree 
FilePath: /compression/Octree.py
All rights reserved.
'''
import numpy as np
from OctreeCPP.Octreewarpper import GenOctree
class CNode():
    def __init__(self,nodeid=0,childPoint=[[]]*8,parent=0,oct=0,pos = np.array([0,0,0]),octant = 0) -> None:
        self.nodeid = nodeid 
        self.childPoint=childPoint.copy() 
        self.parent = parent 
        self.oct = oct # the occupancy symbol of encoding node 1~255
        self.pos = pos 
        self.octant = octant 

class COctree():
    def __init__(self,node=[],level=0) -> None:
        self.node = node.copy()
        self.level=level

def dec2bin(n, count=8): 
    """returns the binary of integer n, using count number of digits""" 
    return [int((n >> y) & 1) for y in range(count-1, -1, -1)]

def dec2binAry(x, bits):
    # 将一维数组 x 中的每个元素转换为固定长度的二进制表示
    mask = np.expand_dims(2**np.arange(bits-1,-1,-1),1).T              
    return (np.bitwise_and(np.expand_dims(x,1), mask)!=0).astype(int) 

def bin2decAry(x): 
    # 将二进制数组转换为十进制整数
    if(x.ndim==1):
        x = np.expand_dims(x,0)
    bits = x.shape[1]
    mask = np.expand_dims(2**np.arange(bits-1,-1,-1),1)
    return x.dot(mask).astype(int)

def Morton(A):
    # 计算三维坐标的 Morton 编码
    A =  A.astype(int)
    n = np.ceil(np.log2(np.max(A)+1)).astype(int)   
    x = dec2binAry(A[:,0],n)                         
    y = dec2binAry(A[:,1],n)
    z = dec2binAry(A[:,2],n)
    m = np.stack((x,y,z),2)                           
    m = np.transpose(m,(0, 2, 1))                     
    mcode = np.reshape(m,(A.shape[0],3*n),order='F')  
    return mcode
 
def DeOctree(LayerCodes, parent_coords, layer_num, Lmax):
    """
    LayerCodes (numpy.ndarray):  父节点的八叉树编码 (十进制数组).
    parent_coords (list of numpy.ndarray): 上一层父节点的坐标列表.
    返回对应子节点的octant coordinate
    """
    occupancyCode = np.flip(dec2binAry(LayerCodes,8),axis=1)

    childNode = [] 
    childOctants = []
    
    code = occupancyCode[0]

    # 占用索引
    for bit in np.where(code==1)[0].tolist():
        newnode = parent_coords + (np.array(dec2bin(bit, count=3)) << (Lmax - layer_num))
        childNode.append(newnode)
        childOctants.append(bit)
        # print(f"Lmax-layernum:{Lmax - layer_num} NodeID:{bit} childNode:{newnode} parentNode:{parent_coords} occupancyCode:{LayerCodes}")

    points = np.array(childNode)
    octants = np.array(childOctants) + 1 # Octant编号从1开始
    num_nodes = len(points)
    return points, octants,num_nodes

# 生成 K 级父节点序列
def GenKparentSeq(Octree, K):
    LevelNum = len(Octree)  
    nodeNum = Octree[-1].node[-1].nodeid 
    Seq = np.ones((nodeNum, K), 'int') * 255  
    LevelOctant = np.zeros((nodeNum, K, 2), 'int')  
    Pos = np.zeros((nodeNum, K, 3), 'int')  
    ChildID = [[] for _ in range(nodeNum)]  
    Seq[0, K-1] = Octree[0].node[0].oct  
    LevelOctant[0, K-1, 0] = 1  
    LevelOctant[0, K-1, 1] = 1  
    Pos[0, K-1, :] = Octree[0].node[0].pos  
    Octree[0].node[0].parent = 1  
    n = 1
    for L in range(1,LevelNum): 
        for node in Octree[L].node: 
            Seq[n, K-1] = node.oct 
            Seq[n, 0:K-1] = Seq[node.parent-1, 1:K]

            LevelOctant[n, K-1, :] = [L+1, node.octant] 
            LevelOctant[n, 0:K-1, :] = LevelOctant[node.parent-1, 1:K, :]
            Pos[n, K-1] = node.pos 
            Pos[n, 0:K-1, :] = Pos[node.parent-1, 1:K, :]
            n += 1 
    assert n == nodeNum  
    DataStruct = {'Seq': Seq, 'Level': LevelOctant, 'ChildID': ChildID, 'Pos': Pos}
    return DataStruct

