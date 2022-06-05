import numpy as np
from glob import glob

import sys
sys.path.append('/cosma/home/dp004/dc-armi2/pywrap_HOD_fitting/src/')
from hod import *
from tpcf_obs import *
from chi2 import *

from mpi4py import MPI
comm =MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

M=sys.argv[1]#'GR'
Lbox = 768
V_survey = 419757894.7368421#Volume from survey
haloes_table = h5py.File('/cosma7/data/dp004/dc-armi2/HOD_mocks/halo_catalogues/Haloes_MG-Gadget_'+M+'_z0.3_L'+str(Lbox)+'_ID_M200c_R200c_pos_vel_Nsh_FirstSh_SubHaloList_SubHaloMass_SubHaloVel_logMmin_11.2.0.hdf5','r')

haloes = np.array(haloes_table['MainHaloes'])
subhaloes = np.array(haloes_table['SubHaloes'])

thetas = np.load('/cosma7/data/dp004/dc-armi2/mcmc_runs/'+M+'_An0.5_Awp0.5/confidence_interval/HODparams_DeltaChi2_1sigma_ConfInt_.npy')

theta_randoms = thetas[np.random.randint(low=0,high=len(thetas),size=1000)]
                                
theta_split = np.array_split(theta_randoms,size)

theta_chunk = theta_split[rank]
n_chunk = np.zeros(len(theta_chunk))
wp_chunk = np.zeros((len(theta_chunk),13))
for i,t in enumerate(theta_chunk):
    G0 = HOD_mock_subhaloes(t,haloes,subhaloes,Lbox)
    wp_sim = wp_from_box(G0,n_threads=28,Lbox=Lbox,Nsigma=13,return_rpavg=False)
    n_sim = G0.shape[0]/Lbox**3
    n_chunk[i] = n_sim
    wp_chunk[i] = wp_sim
    
comm.barrier()
n_all = comm.gather(n_chunk,root=0)
wp_all = comm.gather(wp_chunk,root=0)
theta_all = comm.gather(theta_chunk,root=0)

if rank == 0:
    n_all = np.concatenate(n_all)
    wp_all = np.concatenate(wp_all)
    theta_all = np.concatenate(theta_all)
    T = h5py.File('/cosma7/data/dp004/dc-armi2/HOD_mocks/galaxy_catalogues/boxes/family/datacube/'+M+'_0.5An_0.5Awp_chi_2_1sigma_HODs_wp_ngal.hdf5','w')
    T.create_dataset('theta',data=theta_all)
    T.create_dataset('n_gal',data=n_all)
    T.create_dataset('wp_rp',data=wp_all)
    T.close()
