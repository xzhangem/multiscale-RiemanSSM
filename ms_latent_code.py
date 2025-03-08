import sys
import numpy as np
import SRNF_match
import H2_stats as stats
import H2_ivp as gi
import utils.input_output as io
import plotly.graph_objects as go
import os
import random
from H2_match import H2StandardIterative, H2MultiRes, H2StandardIterative
import torch
import trimesh
import H2_match_coeffs as gmc
import enr.H2 as energy
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='multiscale training of shape latent codes')
    parser.add_argument('--mean_file', type=str, default='./pancreas_001.ply',
            help='file name of mean mesh')
    parser.add_argument('--TPCA_eigvec_prefix', type=str, default='pancreas_evector_272',
            help='prefix of TPCA eigenvector group')
    parser.add_argument('--resolution', type=int, default=3,
            help='scale number for multiscale training')
    parser.add_argument('--a0', type=float, default=1,
            help='a0 coeffecient')
    parser.add_argument('--a1', type=float, default=200,
            help='a1 coeffecient')
    parser.add_argument('--b1', type=float, default=0,
            help='b1 coeffecient')
    parser.add_argument('--c1', type=float, default=200,
            help='c1 coeffecient')
    parser.add_argument('--d1', type=float, default=0,
            help='d1 coeffecient')
    parser.add_argument('--a2', type=float, default=200,
            help='a2 coeffecient')
    parser.add_argument('--geods_len', type=int, default=2,
            help='numbers of state for approximating geodesics')
    parser.add_argument('--input', type=str, default='./pancreas_080.ply',
            help='input file of mesh')
    parser.add_argument('--save_result', action="store_true",
            help='activate to save reults')
    parser.add_argument('--recon_dir', type=str, default=None,
            help='save dir for reconstructed mesh')
    parser.add_argument('--lc_dir', type=str, default=None,
            help='save dir for latent codes')

    param1_sig = { 'weight_coef_dist_T': 10**1,'weight_coef_dist_S': 2*10**1,'sig_geom': 20,'max_iter': 5000,'time_steps': 2, 'kernel_fun':"gaussian", 'sig_fun':0.2, 'tri_unsample': True, 'index':0}
    param2_sig = { 'weight_coef_dist_T': 10**1,'weight_coef_dist_S': 2*10**1,'sig_geom': 10,'max_iter': 5000,'time_steps': 2, 'kernel_fun':"gaussian", 'sig_fun':0.1, 'tri_unsample': False, 'index':1}
    param3_sig = { 'weight_coef_dist_T': 10**1,'weight_coef_dist_S': 2*10**1,'sig_geom': 10,'max_iter': 5000,'time_steps': 2, 'kernel_fun':"gaussian", 'sig_fun':0.075, 'tri_unsample': True, 'index':1}
    param4_sig = { 'weight_coef_dist_T': 10**2,'weight_coef_dist_S': 2*10**2,'sig_geom': 4,'max_iter': 5000,'time_steps': 2, 'kernel_fun':"gaussian", 'sig_fun':0.050, 'tri_unsample': False, 'index':2}
    param5_sig = { 'weight_coef_dist_T': 10**2,'weight_coef_dist_S': 2*10**2,'sig_geom': 4,'max_iter': 5000,'time_steps': 2, 'kernel_fun':"gaussian", 'sig_fun':0.025, 'tri_unsample': False, 'index':2}

    torchdeviceId = torch.device('cuda:0')# if use_cuda else 'cpu'
    torchdtype = torch.float32

    paramlist_sig = [param1_sig, param2_sig, param3_sig, param4_sig, param5_sig]
    args = parser.parse_args()

    template_mesh = trimesh.load(args.mean_file)
    template_v, template_f = np.array(template_mesh.vertices), np.array(template_mesh.faces)
    for i in range(args.resolution-1):
        [template_v, template_f] = io.decimate_mesh(template_v, template_f, int(template_f.shape[0]/4))
    for i in range(args.resolution-1):
        [template_v, template_f] = io.subdivide_mesh(template_v, template_f, order=1)
    template_v = torch.from_numpy(template_v).to(dtype=torchdtype, device=torchdeviceId)
    template_f = torch.from_numpy(template_f).to(dtype=torch.long, device=torchdeviceId)
    template_mesh = [template_v.cpu().numpy(), template_f.cpu().numpy()]

    basis_list = []
    for i in range(args.resolution):
        basis_np = np.load(args.TPCA_eigvec_prefix + '_d' + str(args.resolution-1-i) + '.npy')
        basis_np = torch.from_numpy(basis_np).to(dtype=torchdtype, device=torchdeviceId)
        basis_list.append(basis_np)

    input_mesh = trimesh.load(args.input)
    input_v, input_f = np.array(input_mesh.vertices), np.array(input_mesh.faces)
    #input_mesh = [torch.from_numpy(input_v).to(dtype=torchdtype, device=torchdeviceId), torch.from_numpy(input_f).to(dtype=torchdtype, device=torchdeviceId)]

    input_mesh = [input_v, input_f]
    init_path = torch.stack([template_v]*args.geods_len, dim=0)
    
    chemin_exp, X, F_output = gmc.True_H2MultiRes_sym_coeff(args.a0, args.a1, args.b1, args.c1, args.d1, args.a2, paramlist_sig, template_mesh, input_mesh, init_path, basis_list, args.resolution-1)

    latent_code = X.cpu().numpy() ### Latent code
    V = np.array(chemin_exp[-1].cpu().numpy()) ### Reconstructed vertices
    
    if args.save_result == True:
        save_mesh = trimesh.Trimesh(V, template_f)
        save_mesh.export(args.recon_dir)
        np.save(args.lc_dir, X)

