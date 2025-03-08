# Load Packages
from enr.H2 import *
from H2_param import H2Midpoint
import numpy as np
import scipy
from scipy.optimize import minimize,fmin_l_bfgs_b
from enr.DDG import computeBoundary
from torch.autograd import grad
import utils.utils as io
import torch
use_cuda = 1
torchdeviceId = torch.device('cuda:0') if use_cuda else 'cpu'
torchdtype = torch.float32
import sys
#sys.path.append('../HodgeNet-main/')
from tempfile import TemporaryDirectory
#from HodgeFeature import HodgeFeature
import trimesh


def SymmetricMatching_coeff(a0,a1,b1,c1,d1,a2,param, source, target, chemin, faces, basis):
    sig_geom = param['sig_geom']

    if ('sig_grass' not in param):
        sig_grass = 1
    else:
        sig_grass = param['sig_grass']

    if ('kernel_geom' not in param):
        kernel_geom = 'gaussian'
    else:
        kernel_geom = param['kernel_geom']

    if ('kernel_grass' not in param):
        kernel_grass = 'binet'
    else:
        kernel_grass = param['kernel_grass']

    if ('kernel_fun' not in param):
        kernel_fun = 'constant'
    else:
        kernel_fun = param['kernel_fun']

    if 'sig_fun' not in param:
        sig_fun = 1
    else:
        sig_fun = param['sig_fun']

    #weight_coef_dist = param['weight_coef_dist']
    weight_coef_dist_T=param['weight_coef_dist_T']
    weight_coef_dist_S=param['weight_coef_dist_S']
    max_iter = param['max_iter']
    
    # Convert Data to Pytorch
    VS = torch.from_numpy(source[0]).to(dtype=torchdtype, device=torchdeviceId)
    VT = torch.from_numpy(target[0]).to(dtype=torchdtype, device=torchdeviceId)
    FS = torch.from_numpy(source[1]).to(dtype=torch.long, device=torchdeviceId)
    FT = torch.from_numpy(target[1]).to(dtype=torch.long, device=torchdeviceId)
    F_sol = faces

    N = chemin.shape[0]
    n = chemin.shape[1]

    FunS = torch.from_numpy(np.zeros((int(np.size(source[0]) / 3),))).to(dtype=torchdtype, device=torchdeviceId)
    FunT = torch.from_numpy(np.zeros((int(np.size(target[0]) / 3),))).to(dtype=torchdtype, device=torchdeviceId)
    Fun_sol = torch.from_numpy(np.zeros((int(chemin.shape[1]),))).to(dtype=torchdtype, device=torchdeviceId)

    sig_geom = torch.tensor([sig_geom], dtype=torchdtype, device=torchdeviceId)
    sig_grass = torch.tensor([sig_grass], dtype=torchdtype, device=torchdeviceId)
    sig_fun = torch.tensor([sig_fun], dtype=torchdtype, device=torchdeviceId)

    #chemin = torch.from_numpy(chemin).to(dtype=torchdtype, device=torchdeviceId)
    #chemin = chemin.to(dtype=torchdtype, device=torchdeviceId)
    # Define Energy and set parameters

    energy = enr_match_H2_sym_coeff(VS, FS, FunS, VT, FT, FunT, F_sol, Fun_sol, geod=chemin, basis=basis, weight_coef_dist_T=weight_coef_dist_T, weight_coef_dist_S=weight_coef_dist_S, kernel_geom=kernel_geom,
            kernel_grass=kernel_grass, kernel_fun=kernel_fun, sig_geom=sig_geom, sig_grass=sig_grass, sig_fun=sig_fun, a0=a0, a1=a1, b1=b1, c1=c1, d1=d1, a2=a2)
    
    tm = chemin.shape[0]

    def gradE(X):
        qX = X.clone().requires_grad_(True)
        return grad(energy(qX), qX, create_graph=True)
    

    def funopt(X):
        X=torch.from_numpy(X.reshape(tm,-1)).to(dtype=torchdtype, device=torchdeviceId)
        return float(energy(X).detach().cpu().numpy())

    def dfunopt(X):
        X = torch.from_numpy(X.reshape(tm,-1)).to(dtype=torchdtype, device=torchdeviceId)
        [GX] = gradE(X)
        GX = GX.detach().cpu().numpy().flatten().astype('float64')
        return GX
    
    
    X0 = np.zeros((basis.shape[0]*tm))
    xopt, fopt, Dic = fmin_l_bfgs_b(funopt, X0, fprime=dfunopt, pgtol=1e-05, epsilon=1e-08,
                                    maxiter=max_iter, iprint=1, maxls=20, maxfun=150000,factr=weight_coef_dist_T)
    X_t = torch.from_numpy(xopt.reshape((tm, -1))).to(dtype=torchdtype, device=torchdeviceId)
    chemin_exp = chemin + torch.einsum("ij, jkl-> ikl", X_t, basis)
    return chemin_exp, X_t,fopt, Dic

def StandardMatching_coeff(a0,a1,b1,c1,d1,a2,param, source, target, chemin, faces, basis):    
    sig_geom=param['sig_geom']
    
    if ('sig_grass'not in param):
        sig_grass = 1
    else:
        sig_grass=param['sig_grass']
    
    if ('kernel_geom' not in param):
        kernel_geom = 'gaussian'
    else:
        kernel_geom = param['kernel_geom']
        
    if ('kernel_grass' not in param):
        kernel_grass = 'binet'
    else:
        kernel_grass = param['kernel_grass']
    
    if ('kernel_fun' not in param):
        kernel_fun = 'constant'
    else:
        kernel_fun = param['kernel_fun']
        
    if 'sig_fun' not in param:
        sig_fun = 1
    else:
        sig_fun=param['sig_fun']
        
    weight_coef_dist_T=param['weight_coef_dist_T']
    max_iter=param['max_iter']
    
    # Convert Data to Pytorch
    VS = torch.from_numpy(source[0]).to(dtype=torchdtype, device=torchdeviceId)
    VT = torch.from_numpy(target[0]).to(dtype=torchdtype, device=torchdeviceId)
    FS = torch.from_numpy(source[1]).to(dtype=torch.long, device=torchdeviceId)
    FT = torch.from_numpy(target[1]).to(dtype=torch.long, device=torchdeviceId)
    F_sol = faces

    N = chemin.shape[0]
    n = chemin.shape[1]

    if len(source) == 2:
        FunS = torch.from_numpy(np.zeros((int(np.size(source[0])/3),))).to(dtype=torchdtype, device=torchdeviceId)
        FunT = torch.from_numpy(np.zeros((int(np.size(target[0])/3),))).to(dtype=torchdtype, device=torchdeviceId)
        Fun_sol = torch.from_numpy(np.zeros((int(chemin.shape[1]),))).to(dtype=torchdtype, device=torchdeviceId) #torch.from_numpy(np.zeros((int(geod.shape[1]),))).to(dtype=torchdtype, device=torchdeviceId)
    else:
        FunS = torch.from_numpy(source[2]).to(dtype=torchdtype, device=torchdeviceId)
        FunT = torch.from_numpy(target[2]).to(dtype=torchdtype, device=torchdeviceId)
        Fun_sol = torch.from_numpy(np.zeros((int(chemin.shape[1]),))).to(dtype=torchdtype, device=torchdeviceId)

    #FunS = torch.from_numpy(np.zeros((int(np.size(source[0]) / 3),))).to(dtype=torchdtype, device=torchdeviceId)
    #FunT = torch.from_numpy(np.zeros((int(np.size(target[0]) / 3),))).to(dtype=torchdtype, device=torchdeviceId)
    #Fun_sol = torch.from_numpy(np.zeros((int(chemin.shape[1]),))).to(dtype=torchdtype, device=torchdeviceId)

    sig_geom = torch.tensor([sig_geom], dtype=torchdtype, device=torchdeviceId)
    sig_grass = torch.tensor([sig_grass], dtype=torchdtype, device=torchdeviceId)
    sig_fun = torch.tensor([sig_fun], dtype=torchdtype, device=torchdeviceId)

    #chemin = torch.from_numpy(chemin).to(dtype=torchdtype, device=torchdeviceId)
    chemin = chemin.to(dtype=torchdtype, device=torchdeviceId)
    # Define Energy and set parameters

    energy = enr_match_H2_coeff(VS, VT, FT, FunT, F_sol, Fun_sol, geod=chemin, basis=basis, weight_coef_dist_T=weight_coef_dist_T, kernel_geom=kernel_geom, kernel_grass=kernel_grass, kernel_fun=kernel_fun, sig_geom=sig_geom, sig_grass=sig_grass, sig_fun=sig_fun, a0=a0, a1=a1, b1=b1, c1=c1, d1=d1, a2=a2)
    
    tm = chemin.shape[0]

    def gradE(X):
        qX = X.clone().requires_grad_(True)
        return grad(energy(qX), qX, create_graph=True)
    

    def funopt(X):
        X=torch.from_numpy(X.reshape((tm-1),-1)).to(dtype=torchdtype, device=torchdeviceId)
        return float(energy(X).detach().cpu().numpy())

    def dfunopt(X):
        X = torch.from_numpy(X.reshape((tm-1),-1)).to(dtype=torchdtype, device=torchdeviceId)
        [GX] = gradE(X)
        GX = GX.detach().cpu().numpy().flatten().astype('float64')
        return GX
    
    
    X0 = np.zeros((basis.shape[0]*(tm-1)))
    
    xopt, fopt, Dic = fmin_l_bfgs_b(funopt, X0, fprime=dfunopt, pgtol=1e-05, epsilon=1e-08,
                                    maxiter=max_iter, iprint=1, maxls=20, maxfun=150000,factr=weight_coef_dist_T)
    X_t = torch.from_numpy(xopt.reshape(((tm-1), -1))).to(dtype=torchdtype, device=torchdeviceId)
    X_t=torch.cat((torch.unsqueeze(torch.zeros((X_t.shape[1])).to(dtype=torchdtype, device=torchdeviceId),dim=0),X_t),dim=0)
    chemin_exp = chemin + torch.einsum("ij, jkl-> ikl", X_t, basis)
    return chemin_exp, X_t, fopt, Dic


def H2Midpoint_coeff(geod,faces,newN,a0,a1,b1,c1,d1,a2,param, basis):   
    max_iter=param['max_iter']    
    N=geod.shape[0]

    # Convert Data to Pytorch
    
    if torch.is_tensor(geod):
        geod=geod.cpu().numpy()
    
    xp=np.linspace(0,1,N,endpoint=True)
    x=np.linspace(0,1,newN,endpoint=True)    
    f=scipy.interpolate.interp1d(xp,geod,axis=0)
    geod=f(x)
    
    geod=torch.from_numpy(geod).to(dtype=torchdtype, device=torchdeviceId)
    if not torch.is_tensor(faces):
        F_sol= torch.from_numpy(faces).to(dtype=torch.long, device=torchdeviceId)    
    else:
        F_sol=faces
    n=geod.shape[1]
    
    energy = enr_param_H2_coeff(F_sol,geod,basis,a0,a1,b1,c1,d1,a2)
    
    tm = geod.shape[0]
    
    def gradE(X):
        qX = X.clone().requires_grad_(True)
        return grad(energy(qX), qX, create_graph=True)

    def funopt(X):
        X=torch.from_numpy(X.reshape(tm-2,-1)).to(dtype=torchdtype, device=torchdeviceId)
        return float(energy(X).detach().cpu().numpy())

    def dfunopt(X):
        X = torch.from_numpy(X.reshape(tm-2,-1)).to(dtype=torchdtype, device=torchdeviceId)
        [GX] = gradE(X)
        GX = GX.detach().cpu().numpy().flatten().astype('float64')
        return GX

    X0 = np.zeros((basis.shape[0]*(tm-2)))

    out,fopt,Dic=fmin_l_bfgs_b(funopt, X0, fprime=dfunopt, pgtol=1e-05, epsilon=1e-08, maxiter=max_iter, iprint = 1, maxls=20,maxfun=150000,factr=1e5)
    out=torch.from_numpy(out.reshape(((tm-2), -1))).to(dtype=torchdtype, device=torchdeviceId)
    out=torch.cat((torch.unsqueeze(torch.zeros((out.shape[1])).to(dtype=torchdtype, device=torchdeviceId),dim=0),out,torch.unsqueeze(torch.zeros((out.shape[1])).to(dtype=torchdtype, device=torchdeviceId),dim=0)),dim=0)
    geod = geod + torch.einsum("ij, jkl-> ikl", out, basis)
    return geod, out


def H2Parameterized_coeff(source,target,a0,a1,b1,c1,d1,a2,paramlist,basis):
    F0=source[1]
    geod=np.array([source[0],target[0]])
    for param in paramlist:
        newN= param['time_steps']
        geod,X=H2Midpoint_coeff(geod,F0,newN,a0,a1,b1,c1,d1,a2,param, basis)
        print(geod.shape)
    return geod,X, F0


def H2MultiRes_sym_coeff(a0,a1,b1,c1,d1,a2,paramlist, source, target, chemin, faces, basis):

    total_X= torch.zeros((chemin.shape[0],basis.shape[0])).to(dtype=torchdtype, device=torchdeviceId)
    
    iterations=len(paramlist)
    for j in range(0,iterations):
        params=paramlist[j]
        time_steps= params['time_steps']
        [N,n,three]=chemin.shape
        chemin,X,ener,Dic=SymmetricMatching_coeff(a0,a1,b1,c1,d1,a2,params, source, target, chemin, faces, basis)        
        total_X+=X
        if time_steps>2:
            xp=np.linspace(0,1,total_X.shape[0],endpoint=True)
            x=np.linspace(0,1,time_steps,endpoint=True)    
            f=scipy.interpolate.interp1d(xp,total_X.cpu().numpy(),axis=0)
            total_X=f(x)
            total_X=torch.from_numpy(total_X).to(dtype=torchdtype, device=torchdeviceId)
            
            chemin,X=H2Midpoint_coeff(chemin,faces,time_steps,a0,a1,b1,c1,d1,a2,params, basis)           
            total_X+=X      
    return chemin, total_X


def True_H2MultiRes_sym_coeff(a0, a1, b1, c1, d1, a2, paramlist, source, target, chemin, basis_list, resolutions):
    if len(source) == 2:
        [VS, FS] = source
        [VT, FT] = target
        '''
        sources = [[VS, FS]]
        targets = [[VT, FT]]
        for i in range(0, resolutions):
            [VS,FS]=io.decimate_mesh(VS,FS,int(FS.shape[0]/4))
            sources = [[VS,FS]]+sources
            [VT,FT]=io.decimate_mesh(VT,FT,int(FT.shape[0]/4))
            targets = [[VT,FT]]+targets
        source_init = sources[0]
        target_init = sources[0]
        '''
        for i in range(resolutions):
            [VS, FS] = io.decimate_mesh(VS, FS, int(FS.shape[0]/4))
            [VT, FT] = io.decimate_mesh(VT, FT, int(FT.shape[0]/4))
        sources = [[VS, FS]]
        targets = [[VT, FT]]
        for i in range(resolutions):
            [VS, FS] = io.subdivide_mesh(VS, FS, order=1)
            sources.append([VS, FS])
            [VT, FT] = io.subdivide_mesh(VT, FT, order=1)
            targets.append([VT, FT])
        source_init = sources[0]
        target_init = sources[0]
    else:
        [VS, FS, FunS] = source
        [VT, FT, FunT] = target

        with TemporaryDirectory() as tn:
            for i in range(resolutions):
                [VS, FS] = io.decimate_mesh(VS, FS, int(FS.shape[0]/4))
                [VT, FT] = io.decimate_mesh(VT, FT, int(FT.shape[0]/4))
            io.saveData(file_name=tn+'/source_'+str(resolutions), extension='ply', V=VS, F=FS)
            #source_feature = HodgeFeature(model_path=hodge_model_path, mesh_path=tn+'/source_'+str(resolutions)+'.ply').astype(np.float32)
            #source_feature = np.mean(source_feature, axis=1)
            io.saveData(file_name=tn+'/target_'+str(resolutions), extension='ply', V=VT, F=FT)
            #target_feature = HodgeFeature(model_path=hodge_model_path, mesh_path=tn+'/target_'+str(resolutions)+'.ply').astype(np.float32)
            #target_feature = np.mean(target_feature, axis=1)

            sources = [[VS, FS]]#, source_feature]]
            targets = [[VT, FT]]#, target_feature]]

            for i in range(resolutions):
                [VS, FS] = io.subdivide_mesh(VS, FS, order=1)
                io.saveData(file_name=tn+'/source_'+str(resolutions), extension='ply', V=VS, F=FS)
                #source_feature = HodgeFeature(model_path=hodge_model_path, mesh_path=tn+'/source_'+str(resolutions)+'.ply').astype(np.float32)
                #source_feature = np.mean(source_feature, axis=1)
                sources.append([VS, FS])#, source_feature])

                [VT, FT] = io.subdivide_mesh(VT, FT, order=1)
                io.saveData(file_name=tn+'/target_'+str(resolutions), extension='ply', V=VT, F=FT)
                #target_feature = HodgeFeature(model_path=hodge_model_path, mesh_path=tn+'/target_'+str(resolutions)+'.ply').astype(np.float32)
                #target_feature = np.mean(target_feature, axis=1)
                targets.append([VT, FT])#, target_feature])

        source_init = sources[0]
        target_init = sources[0]
        '''
        with TemporaryDirectory() as tn:
            for i in range(0, resolutions):
                [VS, FS] = io.decimate_mesh(VS, FS, int(FS.shape[0]/4))
                io.saveData(file_name=tn+'/source_'+str(resolutions), extension='ply', V=VS, F=FS)
                source_feature = HodgeFeature(model_path=hodge_model_path, mesh_path=tn+'/source_'+str(resolutions)+'.ply').astype(np.float32)
                source_feature = np.mean(source_feature, axis=1)
                sources = [[VS, FS, source_feature]] + sources

                [VT, FT] = io.decimate_mesh(VT, FT, int(FT.shape[0]/4))
                io.saveData(file_name=tn+'/target_'+str(resolutions), extension='ply', V=VT, F=FT)
                target_feature = HodgeFeature(model_path=hodge_model_path, mesh_path=tn+'/target_'+str(resolutions)+'.ply').astype(np.float32)
                target_feature = np.mean(target_feature, axis=1)
                targets = [[VT, FT, target_feature]] + targets

            source_init = sources[0]
            target_init = sources[0]
        '''

    basis = basis_list[-1]

    total_X= torch.zeros((chemin.shape[0],basis.shape[0])).to(dtype=torchdtype, device=torchdeviceId)

    [N,n,three] = chemin.shape
    chemin = np.zeros((N, source_init[0].shape[0], source_init[0].shape[1]))
    for i in range(N):
        chemin[i,:,:] = source_init[0]

    chemin = torch.from_numpy(chemin).to(dtype=torchdtype, device=torchdeviceId)

    iterations = len(paramlist)
    faces = source_init[1]
    faces = torch.from_numpy(faces).to(dtype=torch.long, device=torchdeviceId)
    for j in range(0, iterations):
        params = paramlist[j]
        time_steps= params['time_steps']
        tri_upsample= params['tri_unsample']
        index= params['index']
        [N,n,three]=chemin.shape
        basis = basis_list[index]
        print((sources[index][0]).shape)
        print((targets[index][0]).shape)
        print(basis.shape)
        print(chemin.shape)
        chemin,X,ener,Dic=SymmetricMatching_coeff(a0,a1,b1,c1,d1,a2,params, sources[index], targets[index], chemin, faces, basis)
        if time_steps > 2:
            xp=np.linspace(0,1,total_X.shape[0],endpoint=True)
            x=np.linspace(0,1,time_steps,endpoint=True)
            f=scipy.interpolate.interp1d(xp,total_X.cpu().numpy(),axis=0)
            total_X=f(x)
            total_X=torch.from_numpy(total_X).to(dtype=torchdtype, device=torchdeviceId)
            chemin,X=H2Midpoint_coeff(chemin,faces,time_steps,a0,a1,b1,c1,d1,a2,params, basis)
        if tri_upsample:
            chemin_sub = []
            faces_sub = []
            for i in range(0,N):
                chemin_cpu = chemin[i].cpu().numpy()
                faces_cpu = faces.cpu().numpy()
                chemin_subi, F_subi = io.subdivide_mesh(chemin_cpu,faces_cpu,order=1)
                faces_sub.append(F_subi)
                chemin_sub.append(chemin_subi)
            chemin_sub = np.stack(chemin_sub, axis=0)
            chemin=torch.from_numpy(chemin_sub).to(dtype=torchdtype, device=torchdeviceId)
            faces = faces_sub[0]
            faces = torch.from_numpy(faces).to(dtype=torch.long, device=torchdeviceId)
            total_X = torch.zeros((chemin.shape[0],basis.shape[0])).to(dtype=torchdtype, device=torchdeviceId)
        else:
            total_X += X

    face_output = faces.cpu().numpy()
    return chemin, total_X, face_output




def H2MultiRes_coeff(a0,a1,b1,c1,d1,a2,paramlist, source, target, chemin, faces, basis):#, hodge_model_path=None):
    if len(source) == 2:
        [VS, FS] = source
        [VT, FT] = target
        source = [VS, FS]
        target = [VT, FT]
    else:
        [VS, FS, FunS] = source
        [VT, FT, FunT] = target
        source = [VS, FS, FunS]
        target = [VT, FT, FunT]

    total_X= torch.zeros((chemin.shape[0],basis.shape[0])).to(dtype=torchdtype, device=torchdeviceId)
    '''
    with TemporaryDirectory() as tn:
        [VS, FS] = source
        [VT, FT] = target
        io.saveData(file_name=tn+'/source_'+str(resolutions), extension='ply', V=VS, F=FS)
        source_feature = HodgeFeature(model_path=hodge_model_path, mesh_path=tn+'/source_'+str(resolutions)+'.ply').astype(np.float32)
        source_feature = np.mean(source_feature, axis=1)
        source = [VS, FS, source_feature]
        io.saveData(file_name=tn+'/target_'+str(resolutions), extension='ply', V=VT, F=FT)
        target_feature = HodgeFeature(model_path=hodge_model_path, mesh_path=tn+'/target_'+str(resolutions)+'.ply').astype(np.float32)
        target_feature = np.mean(target_feature, axis=1)
        target = [VT, FT, target_feature]
    '''
    
    iterations=len(paramlist)
    for j in range(0,iterations):
        params=paramlist[j]
        time_steps= params['time_steps']
        [N,n,three]=chemin.shape
        chemin,X,ener,Dic=StandardMatching_coeff(a0,a1,b1,c1,d1,a2,params, source, target, chemin, faces, basis)        
        total_X+=X
        if time_steps>2:
                
            xp=np.linspace(0,1,total_X.shape[0],endpoint=True)
            x=np.linspace(0,1,time_steps,endpoint=True)    
            f=scipy.interpolate.interp1d(xp,total_X.cpu().numpy(),axis=0)
            total_X=f(x)
            total_X=torch.from_numpy(total_X).to(dtype=torchdtype, device=torchdeviceId)
            
            chemin,X=H2Midpoint_coeff(chemin,faces,time_steps,a0,a1,b1,c1,d1,a2,params, basis)           
            total_X+=X
                      
    return chemin,total_X
