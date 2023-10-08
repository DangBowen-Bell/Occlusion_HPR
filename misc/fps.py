import torch


def farthest_point_sample(xyz, npoint):
    """
    Input: 
        xyz: point cloud data. [B, N, 3]
        npoint: number of samples
    Output:
        centroids: sampled point cloud index. [B, npoint]
    """
    # xyz = xyz.transpose(2,1)
    device = xyz.device
    B, N, C = xyz.shape
    
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)     # 采样点矩阵（B, npoint）
    distance = torch.ones(B, N).to(device) * 1e10                       # 采样点到所有点距离（B, N）

    batch_indices = torch.arange(B, dtype=torch.long).to(device)        # batch_size 数组

    # farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device) # 初始时随机选择一点
    
    barycenter = torch.sum((xyz), 1)                                    #计算重心坐标 及 距离重心最远的点
    barycenter = barycenter/xyz.shape[1]
    barycenter = barycenter.view(B, 1, 3)

    dist = torch.sum((xyz - barycenter) ** 2, -1)
    farthest = torch.max(dist, 1)[1]                                    #将距离重心最远的点作为第一个点

    for i in range(npoint):
        # print("-------------------------------------------------------")
        # print("The %d farthest pts %s " % (i, farthest))
        centroids[:, i] = farthest                                      # 更新第i个最远点
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)        # 取出这个最远点的xyz坐标
        dist = torch.sum((xyz - centroid) ** 2, -1)                     # 计算点集中的所有点到这个最远点的欧式距离
        # print("dist    : ", dist)
        mask = dist < distance
        # print("mask %i : %s" % (i,mask))
        distance[mask] = dist[mask]                                     # 更新distance，记录样本中每个点距离所有已出现的采样点的最小距离
        # print("distance: ", distance)

        farthest = torch.max(distance, -1)[1]                           # 返回最远点索引
 
    return centroids

if __name__ == '__main__':

    # sim_data = torch.rand(3,3,9000)
    sim_data = torch.rand(3,9000,3)
    print(sim_data.dtype)
    print(sim_data)

    centroids = farthest_point_sample(sim_data, 1024)
    
    print("Sampled pts: ", centroids)

