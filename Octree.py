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
        self.nodeid = nodeid # 节点的唯一标识符
        self.childPoint=childPoint.copy() # 子节点的坐标列表
        self.parent = parent # 父节点的索引
        self.oct = oct # the occupancy symbol of encoding node 1~255
        self.pos = pos 
        self.octant = octant # 节点所属的象限编号 1~8

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
 
def DeOctree(Codes):
    # 解析八叉树编码，生成对应的三维点云
    Codes = np.squeeze(Codes)
    occupancyCode = np.flip(dec2binAry(Codes,8),axis=1)  
    codeL = occupancyCode.shape[0]                        
    N = np.ones((30),int) 
    codcal = 0
    L = 0
    while codcal+N[L]<=codeL:
        L +=1
        try:
            N[L+1] = np.sum(occupancyCode[codcal:codcal+N[L],:])
        except:
            assert 0
        codcal = codcal+N[L]
    Lmax = L
    Octree = [COctree() for _ in range(Lmax+1)]
    proot = [np.array([0,0,0])]
    Octree[0].node = proot
    codei = 0
    for L in range(1,Lmax+1):
        childNode = []  # the node of next level
        for currentNode in Octree[L-1].node: # bbox of currentNode
            code = occupancyCode[codei,:]
            for bit in np.where(code==1)[0].tolist():
                newnode =currentNode+(np.array(dec2bin(bit, count=3))<<(Lmax-L))# bbox of childnode
                childNode.append(newnode)
            codei+=1
        Octree[L].node = childNode.copy()
    points = np.array(Octree[Lmax].node)
    return points

# 生成 K 级父节点序列
def GenKparentSeq(Octree, K):
    LevelNum = len(Octree)  # 八叉树的层数
    nodeNum = Octree[-1].node[-1].nodeid  # 计算节点总数
    Seq = np.ones((nodeNum, K), 'int') * 255  # 初始化occupancy symbol
    LevelOctant = np.zeros((nodeNum, K, 2), 'int')  # 记录层级和象限信息
    Pos = np.zeros((nodeNum, K, 3), 'int')  # 记录坐标信息
    ChildID = [[] for _ in range(nodeNum)]  # 存储子节点 ID
    Seq[0, K-1] = Octree[0].node[0].oct  # 初始化根节点
    LevelOctant[0, K-1, 0] = 1  # 根节点的层级
    LevelOctant[0, K-1, 1] = 1  # 根节点的象限
    Pos[0, K-1, :] = Octree[0].node[0].pos  # 根节点的位置
    Octree[0].node[0].parent = 1  # 设置根节点父节点
    n = 0 
    for L in range(LevelNum): # 遍历八叉树的每一层
        for node in Octree[L].node: # 遍历该层的所有节点
            Seq[n, K-1] = node.oct # 当前节点的 occupancyCode 赋值K-1列 K-1列即为自己
            # Seq[node.parent-1, 1:K]记录了node的父节点的K-1个祖辈节点的编码
            # node的父节点的K-1个祖先节点同样为node的父节点
            Seq[n, 0:K-1] = Seq[node.parent-1, 1:K]
            # 记录每个节点的层级 L+1（层号从 1 开始）
            # 记录该节点属于父节点的哪个象限（八叉树有 8 个象限）
            LevelOctant[n, K-1, :] = [L+1, node.octant] # 记录当前节点的层级和象限编号
            LevelOctant[n, 0:K-1, :] = LevelOctant[node.parent-1, 1:K, :]
            Pos[n, K-1] = node.pos # 记录当前节点的空间坐标
            Pos[n, 0:K-1, :] = Pos[node.parent-1, 1:K, :]
            n += 1 # 处理下一个节点
    assert n == nodeNum  # 确保所有节点都被处理
    DataStruct = {'Seq': Seq, 'Level': LevelOctant, 'ChildID': ChildID, 'Pos': Pos}
    return DataStruct

