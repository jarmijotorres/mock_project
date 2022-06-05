import numpy as np
from glob import glob
import pandas as pd
import pyvoro
import sys,h5py

sys.path.append('/cosma/home/dp004/dc-armi2/pywrap_HOD_fitting/src/')
from hod import *
from tpcf_obs import *
from chi2 import *
from astropy.cosmology import Planck15 as cosmo

from mpi4py import MPI
comm =MPI.COMM_WORLD

rank = comm.Get_rank()
size = comm.Get_size()

voro_dir = '/cosma7/data/dp004/dc-armi2/HOD_mocks/galaxy_catalogues/boxes/family/marks/'

M = sys.argv[1]
redshift = sys.argv[2]
Lbox=768
Nslices = 20
ls = Lbox/float(Nslices)
Om0 = cosmo.Om0
Ol0 = 1 - cosmo.Om0
z_snap = 0.3

Hz = 100.0*np.sqrt(Om0*(1.0+z_snap)**3 + Ol0)
Hz_i = (1+z_snap)/Hz

list_i = np.sort(glob('/cosma7/data/dp004/dc-armi2/HOD_mocks/galaxy_catalogues/boxes/family/mocks/Galaxy_'+M+'_'+redshift+'_*'))

list_all = np.array_split(list_i[:112],size)
list_chunk = list_all[rank]

for l in list_chunk:
    cat_name = l.split('/')[-1]
    G1 = np.load(l)
    #G0 = HOD_mock_subhaloes(theta,haloes_table,Lbox=Lbox,weights_haloes=None)  
    vz = G1[:,5]
    z_obs = G1[:,2] + vz*Hz_i
    z_obs[z_obs < 0] += Lbox
    z_obs[z_obs > Lbox] -= Lbox
    G1_obs = np.array([G1[:,0],G1[:,1],z_obs]).T
    
    G1_all = []
    vols_all= []
    for c in range(Nslices):
        zi = c*ls
        zf = (c+1)*ls
        G1_chunk = G1_obs[:,(0,1,2)][(G1_obs[:,2]>zi)&(G1_obs[:,2]<=zf)]

        G1_unique, unique_index, unique_inverse = np.unique(G1_chunk,axis=0,return_index=True,return_inverse=True)
        #is_unique = np.zeros(len(G1_chunk),dtype=int)
            #is_unique[unique_index] = 1  

        V2D = pyvoro.compute_2d_voronoi(points=G1_unique[:,(0,1)],limits=[[0.,Lbox],[0.,Lbox]],dispersion=50.0,periodic=[True,True],)
                #
        voronoi_df = pd.DataFrame(data=V2D)
        #voronoi_df.to_csv(voro_dir+l+'.df.csv')
        vol2D = voronoi_df['volume'].values
        del V2D

        vol2D_chunk = vol2D[unique_inverse]

#        comm.barrier()

        #vols_all = comm.gather(vol2D_chunk,root=0)
        #G1_all = comm.gather(G1_chunk,root=0)
        vols_all.append(vol2D_chunk)
        G1_all.append(G1_chunk)
    #if rank == 0:
    #    print('gathering chunks from ranks...')
    vols_all = np.concatenate(vols_all)
    G1_all = np.concatenate(G1_all,axis=0)
    
    G1_vol = np.vstack([G1_all.T,vols_all]).T
    np.save(voro_dir+'V2D_'+cat_name.split('.npy')[0]+'_%d_slices_ssize%.1lf_sspace.npy'%(Nslices,ls),G1_vol)