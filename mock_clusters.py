import numpy as np
import sys,h5py
from astropy.cosmology import Planck15 as cosmo
from halotools.mock_observables import rp_pi_tpcf
sys.path.append('/cosma/home/dp004/dc-armi2/codes/py_codes/')
sys.path.append('/cosma/home/dp004/dc-armi2/pywrap_HOD_fitting/src/')
from chi2 import *
from binning_data import *

#logMcut = 14.50
Lbox = 768.
haloes_table = h5py.File('/cosma7/data/dp004/dc-armi2/HOD_mocks/halo_catalogues/Haloes_MG-Gadget_GR_z0.3_L%d_ID_M200c_R200c_pos_Nsh_FirstSh_SubHaloList_SubHaloMass_logMmin_11.2.0.hdf5'%Lbox,'r')
MainHaloes = haloes_table['MainHaloes']

G0 = np.loadtxt('/cosma7/data/dp004/dc-armi2/HOD_mocks/galaxy_catalogues/Galaxy_GR_z0.3_L768_HOD_13.102_13.077_13.117_0.150_1.011_pos_HaloM200c_weight.dat')

n_target = np.loadtxt('/cosma7/data/dp004/dc-armi2/Jackknife_runs/JK25_ClusterNumberDensity_z0.2_0.4.txt')# Ngal / Volume survey 0.2 < z < 0.4
n_obs = np.array([n_target[0],0.1*n_target[0]])
wp_target = np.loadtxt('/cosma7/data/dp004/dc-armi2/Jackknife_runs/JK25_wpCG_logrp0.5_50_13bins_pimax80_z0.2_0.4.txt')
wp_obs = np.array([wp_target[:,1]/wp_target[:,0],wp_target[:,2]/wp_target[:,0]]).T
Y_obs= (n_obs,wp_obs)
#
Nsigma = 13
rpbins = np.logspace(np.log10(0.5),np.log10(50),Nsigma+1)
rp = 10**(np.log10(rpbins[:-1]) + np.diff(np.log10(rpbins))[0]/2.)
pi_max = 80
pi_bins = np.arange(0,pi_max+1)
dpi = pi_bins[1] - pi_bins[0]

def cluster_selection_function_Mcut(MainHaloes,Mcut):
    H0 = MainHaloes[MainHaloes['M200c'] > Mcut]
    C0 = np.vstack([H0['pos'].T,H0['M200c']]).T
    xi2D_CG_sim = rp_pi_tpcf(sample1=C0[:,(0,1,2)],sample2=G0[:,(0,1,2)],rp_bins=rpbins,pi_bins=pi_bins,period=Lbox,do_auto=False,do_cross=True,num_threads=16,)
    wp_full_box =  2*np.sum(xi2D_CG_sim,axis=1)*dpi
    n0 = len(C0)/float(Lbox**3)
    n_model = np.array([n0,0.10*n0])
    wp_model = np.array([wp_full_box/rp,0.05*wp_full_box/rp]).T
    return (n_model,wp_model),C0

def cluster_selection_function_lambdaCut(clusters,lambda_0,scatter_lambda):
    lambda_M200c = Richness_mass(clusters['M200c']/cosmo.h,z=0.3,A=A,B=B,C=C,Mpiv=3e14,zpiv=0.18)
    lambda_M200c_scatter = np.e**np.random.normal(loc=np.log(lambda_M200c),scale=scatter_lambda)
    H0 = clusters[lambda_M200c_scatter > lambda_0]
    C0 = np.vstack([H0['pos'].T,H0['M200c']]).T
    xi2D_CG_sim = rp_pi_tpcf(sample1=C0[:,(0,1,2)],sample2=G0[:,(0,1,2)],rp_bins=rpbins,pi_bins=pi_bins,period=Lbox,do_auto=False,do_cross=True,num_threads=16,)
    wp_full_box =  2*np.sum(xi2D_CG_sim,axis=1)*dpi
    n0 = len(C0)/float(Lbox**3)
    n_model = np.array([n0,0.10*n0])
    wp_model = np.array([wp_full_box/rp,0.05*wp_full_box/rp]).T
    return (n_model,wp_model),C0

#in case you do a mass cut only the itereation is over 1 parameter
Mcuts = np.arange(14.44,14.56,0.005)
chis_grid = np.zeros_like(Mcuts)
for i,M in enumerate(Mcuts):
    M=10**M
    Y_model,_ = cluster_selection_function_Mcut(MainHaloes=MainHaloes,Mcut=M)
    chis_grid[i] = chis(y_sim=Y_model,y_obs=Y_obs,A_n=0.5,A_wp=0.5)
    
Mcut = Mcuts[np.argmin(chis_grid)]
Y_model,C0 = cluster_selection_function_Mcut(MainHaloes=MainHaloes,Mcut=10**Mcut)
#Richness mass relation (Capasso et al. 2019)
def Richness_mass(M200c,z,A,B,C,Mpiv=3e14,zpiv=0.18):
    """Mass is divided by little h"""
    return A*(M200c/Mpiv)**B * ((1+z)/(1+zpiv))**C

def Richness_redshift_cut(z):
    return 22*(z/0.15)**0.8

A=38.56
B=0.99
C=-1.13

clusters = MainHaloes[MainHaloes['M200c'] > 5e13]#arbitrary cut to reduce number in catalogue

lambdas = np.arange(50,70,2)
sigmas = np.arange(0.17,0.27,0.01)
# 64, 0.25 GR; 56, 0.23 F5
# 74, 0.26 GR; 66 0.19 F5
chis_grid = np.zeros((len(lambdas),len(sigmas)),dtype=float)
for i in range(len(lambdas)):
    for j in range(len(sigmas)):
        Y_model,_ = cluster_selection_function(clusters=clusters,lambda_0=lambdas[i],scatter_lambda=sigmas[j])
        chis_grid[i,j] = chis(y_sim=Y_model,y_obs=Y_obs,A_n=0.5,A_wp=0.5)

    
#wp_poisson = np.zeros((100,len(rp)))
#n_poisson = np.zeros(100)
#for i in range(100):
#    Y_model,C0 = cluster_selection_function(clusters=clusters,lambda_0=64,scatter_lambda=0.25)
#    n_poisson[i] = Y_model[0][0]
#    wp_poisson[i] =  Y_model[1][:,0]

#Y_model,C0 = cluster_selection_function(clusters=clusters,lambda_0=56,scatter_lambda=0.23)

np.savetxt('/cosma7/data/dp004/dc-armi2/HOD_mocks/cluster_catalogues/Cluster_GR_z0.3_L768_M200c_pos_mainhaloes_richness0_64_sigma_0.25.dat',C0)
#S = np.array([rp,np.mean(wp_poisson,axis=0),np.std(wp_poisson,axis=0)]).T
#np.savetxt('/cosma7/data/dp004/dc-armi2/HOD_mocks/observables/number_density/n_cluster_GR_z0.3_Poisson100_error.dat',np.array([np.mean(n_poisson),np.std(n_poisson)]).T,newline=' ')
#np.savetxt('/cosma7/data/dp004/dc-armi2/HOD_mocks/observables/clustering/Poisson100_wpCG_GR_z0.3_L768_rp_0.5_50_13rpbins.dat',S)