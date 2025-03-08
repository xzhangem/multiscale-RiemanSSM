import argparse
import sys
import numpy as np
import H2_stats as stats
import utils.input_output as io
import os
import random
from H2_match import H2StandardIterative, H2MultiRes
import torch
from varifold_mean_pick import varifold_mean_file
from tempfile import TemporaryDirectory
import trimesh


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Template Generation and T-PCA')
    parser.add_argument('--data_file', type=str,
            help='file name of mesh data')
    parser.add_argument('--pre_align', action="store_true", 
            help='activate to prealign dataset')
    parser.add_argument('--prealign_file', type=str, 
            help='files to save prealign mesh by the template')
    parser.add_argument('--template_save', type=str, default='mean.ply',
            help='filename to save the template')
    parser.add_argument('--resolution', type=int, default=3,
            help='scale number for multiscale training')
    parser.add_argument('--input_mean', type=str, default=None,
            help='use user-specific template for T-PCA, default None')
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
    parser.add_argument('--components_num', type=int, default=2,
            help="PCA component number")
    parser.add_argument('--pca_save_name', type=str, default='pca_test',
            help="PCA eigenvalue/vector save name")

    param1_sig = { 'weight_coef_dist_T': 10**1,'weight_coef_dist_S': 1*10**1,'sig_geom': 10,'max_iter': 30,'time_steps': 2}
    param2_sig = { 'weight_coef_dist_T': 10**1,'weight_coef_dist_S': 1*10**1,'sig_geom': 5,'max_iter': 30,'time_steps': 2}
    param3_sig = { 'weight_coef_dist_T': 10**1,'weight_coef_dist_S': 1*10**1,'sig_geom': 5,'max_iter': 30,'time_steps': 2}
    param4_sig = { 'weight_coef_dist_T': 10**1,'weight_coef_dist_S': 2*10**2,'sig_geom': 2,'max_iter': 30,'time_steps': 2}
    param5_sig = { 'weight_coef_dist_T': 10**1,'weight_coef_dist_S': 2*10**2,'sig_geom': 2,'max_iter': 30,'time_steps': 2}

    paramlist_sig = [param1_sig, param2_sig, param3_sig, param4_sig, param5_sig]

    args = parser.parse_args()
    if args.pre_align and not args.prealign_file:
        parser.error("need to assign a dataset name to store prealign files")

    dataset_dir = os.listdir(args.data_file)
    ### Pick out the one with minimal varifold distance as the initial template
    template_file = varifold_mean_file(mesh_file=args.data_file)
    initial_template = trimesh.load(args.data_file + template_file)

    align_samples = []
    if args.pre_align == True:
        if not os.path.exists(args.prealign_file):
            os.makedirs(args.prealign_file)
        for i in range(len(dataset_dir)):
            print("\n Sample {}/{} prealign with template".format(i+1, len(dataset_dir)-1))
            if dataset_dir[i] != template_file:
                mesh = trimesh.load(args.data_file + dataset_dir[i])
                trans_matrix, cost = trimesh.registration.mesh_other(mesh, initial_template, samples=4000, scale=True, icp_first=20, icp_final=60)
                trans_v = trimesh.transformations.transform_points(mesh.vertices, trans_matrix)
                aligned_mesh = trimesh.Trimesh(vertices=trans_v, faces=mesh.faces)
                aligned_mesh.export(args.prealign_file + dataset_dir[i])
                align_samples.append([np.array(trans_v), np.array(mesh.faces)])

    else:
        for i in range(len(dataset_dir)):
            if dataset_dir[i] != template_file:
                mesh = trimesh.load(args.data_file + dataset_dir[i])
                align_samples.append([np.array(mesh.vertices), np.array(mesh.faces)])

    align_samples.append([np.array(initial_template.vertices), np.array(initial_template.faces)])

    V_mu, F_mu = stats.H2UnparamKMean(align_samples, [np.array(initial_template.vertices), np.array(initial_template.faces)],
        args.a0, args.a1, args.b1, args.c1, args.d1, args.a2, paramlist_sig, N=None, geods_len=args.geods_len)

    print("Karcher Mean Estimation Finished")
    mean_mesh = trimesh.Trimesh(vertices=np.array(V_mu), faces=np.array(F_mu))
    mean_mesh.export(args.template_save)

    if args.input_mean != None:
        mean_mesh = trimesh.load(args.input_mean)

    ### Construct the multiscale dataset
    V_mu, F_mu = np.array(mean_mesh.vertices), np.array(mean_mesh.faces)
    [V, F] = [V_mu, F_mu]
    for i in range(args.resolution-1):
        [V, F] = io.decimate_mesh(V, F, int(F.shape[0]/4))

    mean_mesh_v_up_list = []
    mean_mesh_f_up_list = []
    mean_mesh_v_up_list.append(V)
    mean_mesh_f_up_list.append(F)
    for i in range(1, args.resolution):
        print(i)
        [V, F] = io.subdivide_mesh(mean_mesh_v_up_list[i-1], mean_mesh_f_up_list[i-1], order=1)
        mean_mesh_v_up_list.append(V)
        mean_mesh_f_up_list.append(F)


    multiscale_data_list = []
    for i in range(args.resolution):
        multiscale_data_list.append([])

    for i in range(len(align_samples)):
        mesh_array = align_samples[i]
        mesh_v, mesh_f = mesh_array[0], mesh_array[1]
        for j in range(args.resolution-1):
            [mesh_v, mesh_f] = io.decimate_mesh(mesh_v, mesh_f, int(mesh_f.shape[0]/4))
        multiscale_data_list[0].append([mesh_v, mesh_f])

        for k in range(1, args.resolution):
            [mesh_v, mesh_f] = io.subdivide_mesh(mesh_v, mesh_f, order=1)
            multiscale_data_list[k].append([mesh_v, mesh_f])


    ### Construct multiscale T-PCA representation of the dataset 
    for i in range(args.resolution):
        evalue, evector, PCp_V, PCp_F, PCn_V, PCn_F = stats.H2_UnparamPCA(mean_mesh_v_up_list[i], multiscale_data_list[i], mean_mesh_f_up_list[i], 
                args.a0, args.a1, args.b1, args.c1, args.d1, args.a2, paramlist_sig, components=args.components_num, geods_len=args.geods_len)

        np.save(args.pca_save_name + '_d' + str(args.resolution-1-i) + '.npy', evalue)













