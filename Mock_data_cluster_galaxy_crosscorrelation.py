import numpy as np
from halotools.mock_observables import wp

#theta = np.array([13.107185, 14.0771384 , 13.30738439,  0.11,  1.0]) #best-fit parameters
Lbox = 1536.
G0 = np.loadtxt('/cosma7/data/dp004/dc-armi2/HOD_mocks/galaxy_catalogues/Galaxy_GR_z0.3_L1536_HOD_13.11_14.08_13.31_0.11_1.00_pos_M200c_weight_subhaloespos.dat')
C0 = np.loadtxt('/cosma7/data/dp004/dc-armi2/HOD_mocks/cluster_catalogues/Cluster_GR_z0.3_L1536_M200c_pos_mainhaloes.dat')

wp_CODEX_LOWZ = np.loadtxt('/cosma/home/dp004/dc-armi2/sdsswork/outputs_pipelines/data/wp_CODEX_LOWZ_3Dps_logdim0.1_50_13bins_dim2_0_100_100bins.dat')

Nsigma = 13
sigma_bins = np.linspace(0.1,50,Nsigma+1)
pi_max = 100
pi_bins = np.arange(0,pi_max+1)

wp_CG_sim = wp(sample1=C0[:,(0,1,2)],rp_bins=sigma_bins,pi_max=pi_max,sample2=G0[:,(0,1,2)],period=Lbox,do_auto=False,do_cross=True,num_threads=16)