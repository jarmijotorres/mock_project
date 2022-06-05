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
redshift = float(sys.argv[2])#0.3
Lbox = 768

haloes_table = h5py.File('/cosma7/data/dp004/dc-armi2/HOD_mocks/halo_catalogues/Haloes_MG-Gadget_'+M+'_z'+str(redshift)+'_L'+str(Lbox)+'_ID_M200c_R200c_pos_vel_Nsh_FirstSh_SubHaloList_SubHaloMass_SubHaloVel_logMmin_11.2.0.hdf5','r')

haloes = np.array(haloes_table['MainHaloes'])
subhaloes = np.array(haloes_table['SubHaloes'])

tName = sys.argv[3]
thetas = np.load(tName)

np.random.shuffle(thetas)
Nt = 1000
thetas_randoms = np.round(thetas[:1000],decimals=5)
               
thetas_all = np.array_split(thetas_randoms,size)    

thetas_chunk = thetas_all[rank]

n_chunk = np.zeros(len(thetas_chunk))
wp_chunk = np.zeros((len(thetas_chunk),13))
for i,t in enumerate(thetas_chunk):
    G0 = HOD_mock_subhaloes(t,haloes,subhaloes,Lbox)
    wp_sim = wp_from_box(G0,n_threads=16,Lbox=Lbox,Nsigma=13,return_rpavg=False)
    n_sim = G0.shape[0] / Lbox**3
    np.save('/cosma7/data/dp004/dc-armi2/HOD_mocks/galaxy_catalogues/boxes/family/mocks/Galaxy_{0}_{1}_L768_HOD_{2:.5f}_{3:.5f}_{4:.5f}_{5:.5f}_{6:.5f}_MCMCfit.npy'.format(M,redshift,t[0],t[1],t[2],t[3],t[4]),G0)
    n_chunk[i] = n_sim
    wp_chunk[i] = wp_sim

comm.barrier() 
n_all = comm.gather(n_chunk,root=0)
wp_all = comm.gather(wp_chunk,root=0)

if rank == 0:
    n_all = np.concatenate(n_all)
    wp_all = np.concatenate(wp_all)
    T = h5py.File('/cosma7/data/dp004/dc-armi2/mcmc_runs/'+M+'_An0.5_Awp0.5/confidence_interval/randoms_sample_'+M+'_chi_2_1sigma_ConfInt_'+str(Nt)+'_theta_nga_wp_z'+str(redshift)+'.hdf5','w')
    T.create_dataset(name='theta',data=thetas_randoms)
    T.create_dataset(name='n_gal',data=n_all)
    T.create_dataset(name='wp_rp',data=wp_all)
    T.close()