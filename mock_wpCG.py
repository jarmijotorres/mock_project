import numpy as np
#import sys,h5py
from halotools.mock_observables import wp_jackknife
from halotools.mock_observables import rp_pi_tpcf

#theta = #np.array([13.30769231, 14.04615385, 13.45      ,  0.25384615,  0.95]) 
Lbox = 768
C0 = np.loadtxt('/cosma7/data/dp004/dc-armi2/HOD_mocks/cluster_catalogues/Cluster_GR_z0.3_L768_M200c_pos_mainhaloes_logMcut')

G0 = np.loadtxt('/cosma7/data/dp004/dc-armi2/HOD_mocks/galaxy_catalogues/Galaxy_GR_z0.3_L768_HOD_13.10_14.08_13.12_0.11_1.01_pos_M200c_weight.dat')

Nsigma = 13
rpbins = np.logspace(np.log10(0.5),np.log10(50),Nsigma+1)
rp = 10**(np.log10(rpbins[:-1]) + np.diff(np.log10(rpbins))[0]/2.)
pi_max = 80
pi_bins = np.arange(0,pi_max+1)
dpi = pi_bins[1] - pi_bins[0]

R0 = Lbox*np.random.random(size=(len(G0),3))

Nsub=3
wp_JK = wp_jackknife(sample1=C0[:,(0,1,2)],randoms=R0,rp_bins=rpbins,pi_max=pi_max,Nsub=[Nsub,Nsub,Nsub],sample2=G0[:,(0,1,2)],do_auto=False,do_cross=True,estimator='Landy-Szalay',num_threads=16,) 

JK_mean = wp_JK[0]
JK_std = np.array(np.sqrt(wp_JK[1].diagonal())).reshape(-1)

xi2D_CG_sim = rp_pi_tpcf(sample1=C0[:,(0,1,2)],sample2=G0[:,(0,1,2)],rp_bins=rpbins,pi_bins=pi_bins,period=Lbox,do_auto=False,do_cross=True,num_threads=16,)
wp_full_box =  2*np.sum(xi2D_CG_sim,axis=1)*dpi

S = np.array([rp,wp_full_box,JK_std]).T
np.savetxt('/cosma7/data/dp004/dc-armi2/HOD_mocks/observables/clustering/JK27_wpCG_higherrichness_GR_z0.3_L768_rp_0.5_50_13rpbins.dat',S)
