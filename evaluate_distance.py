import sys
sys.path.append('../mesh2SSM_2023-main/')
import os
import torch
from torch.utils.data import DataLoader
import sklearn.metrics as metrics
from chamfer_distance import ChamferDistance
#from pytorch3d.loss import chamfer_distance
from metrics import *
from data import Meshes, MeshesWithFaces
import trimesh
import numpy as np
from icp import ICP
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
import ot
from scipy.stats import wasserstein_distance
criterion = ChamferDistance()


def emd(x, y):
    x_num = x.shape[0]
    y_num = y.shape[0]
    x_vec = np.squeeze(np.ones((x_num,1)) / x_num)
    y_vec = np.squeeze(np.ones((y_num,1)) / y_num)
    print(x_vec.shape)
    print(y_vec.shape)
    d = cdist(x, y)
    s_emd = ot.sinkhorn2(x_vec, y_vec, d, 0.025, verbose=False)
    print(s_emd)
    return s_emd
    #return ot.emd2(x_vec, y_vec, d)

use_cuda = False
device = torch.device("cuda" if use_cuda else "cpu")


result_file =  # result filename
test_file = # ground truth filename

result_dir = os.listdir(result_file)
test_dir = os.listdir(test_file)

mode = result_dir[0].split('.')[-1]
use_scale = False

iterative_closest_points = ICP(False)
iterative_closest_points.to(device)

if use_scale == False:
    scale = 1.0
else:
    scale = 0
    for i in range(len(test_dir)):
        mesh = trimesh.load(test_file + test_dir[i])
        mesh_v = np.array(mesh.vertices)
        if np.max(np.abs(mesh_v)) > scale:
            scale = np.max(np.abs(mesh_v))



chamfer_dist = []
emd_dist = []
for i in range(len(result_dir)):
    test_v = trimesh.load(test_file + test_dir[i])
    test_v_np = np.array(test_v.vertices)
    print(test_v_np.shape)
    #test_v = np.expand_dims(test.vertices, 0)
    test_v = torch.from_numpy(test_v.vertices).unsqueeze(0).to(device).to(dtype=torch.float32)
    #test_v = test_v / scale
    if mode == 'ply':
        result_v = trimesh.load(result_file + result_dir[i])
        result_v = trimesh.smoothing.filter_humphrey(result_v, alpha=0.1, beta=0.1, iterations=10)
        #result_v = np.expand_dims(result.vertices, 0)
        result_v = torch.from_numpy(result_v.vertices).unsqueeze(0).to(device).to(dtype=torch.float32)
    elif mode == 'obj':
        result_v = trimesh.load(result_file + test_dir[i].split('.')[0] + '_deformed.obj')
        result_v = trimesh.smoothing.filter_humphrey(result_v, alpha=0.1, beta=0.1, iterations=10)
        #result_v = np.expand_dims(result.vertices, 0)
        result_v = torch.from_numpy(result_v.vertices).unsqueeze(0).to(device).to(dtype=torch.float32)
    else:
        result_v = np.loadtxt(result_file+result_dir[i])
        #result_v = np.expand_dims(result_v, 0)
        result_v = torch.from_numpy(result_v).unsqueeze(0).to(device).to(dtype=torch.float32)

    #result_v = result_v / scale
    align_result_v = iterative_closest_points(result_v, test_v)
    test_v = test_v / scale
    align_result_v = align_result_v / scale
    test_v_np = test_v.cpu().numpy()
    align_result_np = align_result_v.cpu().numpy()
    emd_dist.append(emd(np.squeeze(test_v_np), np.squeeze(align_result_np)))

    dist1, dist2, idx1, idx2 = criterion(test_v, align_result_v)
    loss = 0.5 * (dist1.sqrt().mean() + dist2.sqrt().mean())
    print(loss.detach().item())
    chamfer_dist.append(loss.detach().item())


print(f'Testing Chamfer Dist: {np.mean(chamfer_dist)} +/- {np.std(chamfer_dist)}')
print(f'Testing EMD Dist: {np.mean(emd_dist)} +/- {np.std(emd_dist)}')

p2m_dist = []

for i in range(len(result_dir)):
    test = trimesh.load(test_file + test_dir[i])
    if mode == 'ply':
        result = trimesh.load(result_file + result_dir[i])
        result = trimesh.smoothing.filter_humphrey(result, alpha=0.1, beta=0.1, iterations=10)
        result_v = np.array(result.vertices)
    elif mode == 'obj':
        result = trimesh.load(result_file + test_dir[i].split('.')[0] + '_deformed.obj')
        result = trimesh.smoothing.filter_humphrey(result, alpha=0.1, beta=0.1, iterations=10)
        result_v = np.array(result.vertices)
    else:
        result_v = np.loadtxt(result_file + result_dir[i])

    test_v = torch.from_numpy(test.vertices).unsqueeze(0).to(device).to(dtype=torch.float32)
    result_v_torch = torch.from_numpy(result_v).unsqueeze(0).to(device).to(dtype=torch.float32)
    align_result_v = iterative_closest_points(result_v_torch, test_v)
    result_v = align_result_v.squeeze().cpu().numpy()
    #print(result_v.shape)

    c = trimesh.proximity.ProximityQuery(test)
    p2mDist = c.signed_distance(result_v)
    p2mDist = np.abs(p2mDist)

    p2mDist = np.mean(p2mDist)
    p2m_dist.append(p2mDist)
    print(p2mDist)

print(f'Testing Point to Mesh Dist: {np.mean(p2m_dist)} +/- {np.std(p2m_dist)}')

