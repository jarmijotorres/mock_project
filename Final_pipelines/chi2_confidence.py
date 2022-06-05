import numpy as np
from glob import glob
#from astropy.table import Table
#from scipy.stats import binned_statistic_2d
#from scipy.ndimage import zoom
#from scipy.ndimage.filters import gaussian_filter
#from scipy.stats import chi2

#import sys
#sys.path.append('/cosma/home/dp004/dc-armi2/pywrap_HOD_fitting/src/')
#from hod import *
#from tpcf_obs import *
#from chi2 import *

#from mpi4py import MPI
#comm =MPI.COMM_WORLD
#rank = comm.Get_rank()
#size = comm.Get_size()

M=sys.argv[1]#'GR'
Lbox = 768
#haloes_table = h5py.File('/cosma7/data/dp004/dc-armi2/HOD_mocks/halo_catalogues/Haloes_MG-Gadget_'+M+'_z0.3_L'+str(Lbox)+'_ID_M200c_R200c_pos_vel_Nsh_FirstSh_SubHaloList_SubHaloMass_SubHaloVel_logMmin_11.2.0.hdf5','r')

#haloes = np.array(haloes_table['MainHaloes'])
#subhaloes = np.array(haloes_table['SubHaloes'])

An = sys.argv[2]#0.4
Awp = sys.argv[3]#0.6

samples = []
lklhd_samples = []
for i in range(2,29):
    file_chain = '/cosma7/data/dp004/dc-armi2/mcmc_runs/GR_An0.5_Awp0.5/chains/MCMCpost_chains_HOD_GR_L768_500it_28walkers_0.5An_0.5Awp_target_CMASS_z0.46_0.54_err_1sigma_fullcov_sim_subhaloes_batch_%d.npy'%i

    file_likelihood = '/cosma7/data/dp004/dc-armi2/mcmc_runs/GR_An0.5_Awp0.5/likelihoods/MCMClklhd_chains_HOD_GR_L768_500it_28walkers_0.5An_0.5Awp_target_CMASS_z0.46_0.54_err_1sigma_fullcov_sim_subhaloes_batch_%d.npy'%i
    
    chains = np.load(file_chain)
    loglikelihood = np.load(file_likelihood)
    samples.append(chains.reshape(chains.shape[0]*chains.shape[1],chains.shape[2]))
    lklhd_samples.append(loglikelihood.T.flatten())
samples = np.concatenate(samples)
lklhd_samples = np.concatenate(lklhd_samples)

max_likelihood = lklhd_samples.max()
chi_2_samples = -2*(lklhd_samples)
chi_min = -2*max_likelihood
Delta_chi_square = chi_2_samples - chi_min

#the value for 1sigma CI using a chi^2 distribution with 5 dof is 5.89
theta_1s = samples[(Delta_chi_square<5.89)]
theta_1s_unique = np.unique(theta_1s,axis=0)
#theta_randoms = theta_1s_unique[np.random.randint(low=0,high=len(theta_1s_unique),size=1000)]

np.save('/cosma7/data/dp004/dc-armi2/mcmc_runs/'+M+'_An'+str(An)+'_Awp'+str(Awp)+'/confidence_interval/HODparams_DeltaChi2_1sigma_ConfInt.npy',theta_1s_unique)
