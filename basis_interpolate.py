import sys
import numpy as np
import os 
from H2_match import H2StandardIterative, H2MultiRes, H2StandardIterative
import torch
import trimesh
import H2_match_coeffs as gmc
import enr.H2 as energy
import trimesh
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="shape interpolation with latent codes")
    parser.add_argument('--mean_file', type=str, help='file name or mean mesh')
    parser.add_argument('--TPCA_eigvec', type=str, help='TPCA eigenvector file, ends with .np')
    parser.add_argument('--source_mesh', type=str, help='source mesh file')
    parser.add_argument('--target_mesh', type=str, help='target mesh file')
    parser.add_argument('--source_code', type=str, help='source mesh latent code file, ends with .np')
    parser.add_argument('--target_code', type=str, help='target mesh latent code file, ends with .np')
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
    parser.add_argument('--intplt_num', type=int, default=2,
            help='interpolate state number')
    parser.add_argument('--save_file', type=str, default='./interplote_result/', help='file to save the interpolated results')


    torchdeviceId = torch.device('cuda:0')# if use_cuda else 'cpu'
    torchdtype = torch.float32

    paramlist_sig = [param1_sig, param2_sig, param3_sig, param4_sig, param5_sig]
    args = parser.parse_args()
    param_sig = { 'weight_coef_dist_T': 10**1,'weight_coef_dist_S': 2*10**1,'sig_geom': 4,'max_iter': 1000,'time_steps': 2, 'kernel_fun':"gaussian", 'sig_fun':0.1 }
    paramlist_sig = []
    for i in range(args.intplt_num-1):
        param_sig['time_steps'] = 2 + i
        paramlist_sig.append(param_sig)

    source_mesh = trimesh.load(args.source_mesh)
    target_mesh = trimesh.load(args.target_mesh)

    source_coeff = np.load(args.source_code)
    target_coeff = np.load(args.target_code)

    source_coeff = torch.from_numpy(source_coeff).to(dtype=torchdtype, device=torchdeviceId)
    target_coeff = torch.from_numpy(target_coeff).to(dtype=torchdtype, device=torchdeviceId)

    template_mesh = trimesh.load(args.mean_file)
    template_v, template_f = np.array(template_mesh.vertices), np.array(template_mesh.faces)
    for i in range(args.resolution-1):
        [template_v, template_f] = io.decimate_mesh(template_v, template_f, int(template_f.shape[0]/4))
    for i in range(args.resolution-1):
        [template_v, template_f] = io.subdivide_mesh(template_v, template_f, order=1)
    template_v = torch.from_numpy(template_v).to(dtype=torchdtype, device=torchdeviceId)
    template_f = torch.from_numpy(template_f).to(dtype=torch.long, device=torchdeviceId)
    template_mesh = [template_v.cpu().numpy(), template_f.cpu().numpy()]

    source_mesh = [np.array(source_mesh.vertices), np.array(target_mesh.faces)]
    target_mesh = [np.array(target_mesh.vertices), np.array(target_mesh.faces)]

    init_path = torch.stack([template_v]*2,dim=0)
    init_X=torch.stack([source_coeff[0], target_coeff[0]],dim=0).to(dtype=torchdtype, device=torchdeviceId)
    init_chemin = init_path + torch.einsum("ij, jkl-> ikl",init_X, basis_torch)

    chemin_exp_output,X= gmc.H2MultiRes_sym_coeff(a0,a1,b1,c1,d1,a2,paramlist_sig, source, target, init_chemin, faces, basis_torch)

    if not os.path.exists(args.save_file):
        os.makedirs(args.save_file)

    frame_n = chemin_exp_output.shape[0]
    for i in range(0, frame_n):
        mesh = trimesh.Trimesh(vertices=np.array(chemin_exp_output[i].cpu().numpy()), faces=np.array(faces.cpu().numpy()))
        mesh = trimesh.smoothing.filter_humphrey(mesh, alpha=0.1, beta=0.1, iterations=3)
        mesh.export(args.save_file + 'interploate_' + str(i) + '.ply')










