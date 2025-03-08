import sys
sys.path.append('..')
import numpy as np
import torch 
from enr.varifold import VKerenl, lossVarifoldSurf
import utils.input_output as io 
import os 
from scipy.integrate import simps
import math
import matplotlib.pyplot as ply

use_cuda = 1
torchdeviceId = torch.device('cuda:0') if use_cuda else 'cpu'
torchdtype = torch.float32

def varifold_mean_file(mesh_file, sig_grass=1.0, kernel_geom='gaussian', kernel_grass='binet', kernel_fun='constant', sig_fun=1.0, sig_geom=5.0):
    mesh_dir = os.listdir(mesh_file)
    mesh_len = len(mesh_dir)
    varifold_matrix = np.zeros((mesh_len, mesh_len))

    sig_grass = torch.tensor([sig_grass], dtype=torchdtype, device=torchdeviceId)
    sig_fun = torch.tensor([sig_fun], dtype=torchdtype, device=torchdeviceId)
    sig_geom = torch.tensor([sig_geom], dtype=torchdtype, device=torchdeviceId)
    K = VKerenl(kernel_geom, kernel_grass, kernel_fun, sig_geom, sig_grass, sig_fun)
    for i in range(mesh_len):
        [V_i, F_i, Fun_i] = io.loadData(mesh_file + mesh_dir[i])
        V_i = V_i / 10
        mesh_i = [np.array(V_i), np.array(F_i)]
        V_i = torch.from_numpy(V_i).to(dtype=torchdtype, device=torchdeviceId)
        F_i = torch.from_numpy(F_i).to(dtype=torch.long, device=torchdeviceId)
        Fun_i = torch.from_numpy(np.zeros((int(np.size(mesh_i[0]/3)),))).to(dtype=torchdtype, device=torchdeviceId)
        for j in range(i, mesh_len):
            [V_j, F_j, Fun_j] = io.loadData(mesh_file + mesh_dir[j])
            V_j = V_j / 10
            mesh_j = [np.array(V_j), np.array(F_j)]
            V_j = torch.from_numpy(V_j).to(dtype=torchdtype, device=torchdeviceId)
            F_j = torch.from_numpy(F_j).to(dtype=torch.long, device=torchdeviceId)
            Fun_j = torch.from_numpy(np.zeros((int(np.size(mesh_j[0]/3)),))).to(dtype=torchdtype, device=torchdeviceId)
            varifold_ij = lossVarifoldSurf(F_i, Fun_i, V_j, F_j, Fun_j, K)(V_i)
            varifold_matrix[i][j] = varifold_ij.cpu().numpy()
            varifold_matrix[j][i] = varifold_ij.cpu().numpy()
        varifold_row_sum = varifold_matrix.sum(axis=1)

        min_indx = np.argmin(varifold_row_sum)
        return mesh_dir[min_indx]
'''
mesh_file = '../Thyroid_ShapeNet/ThyroidRight_ShapeNet/'
mesh_dir = os.listdir(mesh_file)
mesh_len = len(mesh_dir)

varifold_matrix = np.zeros((mesh_len, mesh_len))

sig_grass = 1.0
kernel_geom = 'gaussian'
kernel_grass = 'binet'
kernel_fun = 'constant'
sig_fun = 1.0

sig_geom = 5.0
sig_grass = torch.tensor([sig_grass], dtype=torchdtype, device=torchdeviceId)
sig_fun = torch.tensor([sig_fun], dtype=torchdtype, device=torchdeviceId)
sig_geom = torch.tensor([sig_geom], dtype=torchdtype, device=torchdeviceId)

K = VKerenl(kernel_geom, kernel_grass, kernel_fun, sig_geom, sig_grass, sig_fun)

for i in range(mesh_len):
    print(i)
    [V_i, F_i, Fun_i] = io.loadData(mesh_file + mesh_dir[i])
    V_i = V_i / 10
    mesh_i = [np.array(V_i), np.array(F_i)]
    V_i = torch.from_numpy(V_i).to(dtype=torchdtype, device=torchdeviceId)
    F_i = torch.from_numpy(F_i).to(dtype=torch.long, device=torchdeviceId)
    Fun_i = torch.from_numpy(np.zeros((int(np.size(mesh_i[0]/3)),))).to(dtype=torchdtype, device=torchdeviceId)
    for j in range(i, mesh_len):
        [V_j, F_j, Fun_j] = io.loadData(mesh_file + mesh_dir[j])
        V_j = V_j / 10
        mesh_j = [np.array(V_j), np.array(F_j)]
        V_j = torch.from_numpy(V_j).to(dtype=torchdtype, device=torchdeviceId)
        F_j = torch.from_numpy(F_j).to(dtype=torch.long, device=torchdeviceId)
        Fun_j = torch.from_numpy(np.zeros((int(np.size(mesh_j[0]/3)),))).to(dtype=torchdtype, device=torchdeviceId)
        varifold_ij = lossVarifoldSurf(F_i, Fun_i, V_j, F_j, Fun_j, K)(V_i)

        varifold_matrix[i][j] = varifold_ij.cpu().numpy()
        varifold_matrix[j][i] = varifold_ij.cpu().numpy()


varifold_row_sum = varifold_matrix.sum(axis=1)

min_indx = np.argmin(varifold_row_sum)
print(min_indx)
print(mesh_dir[min_indx])
'''
