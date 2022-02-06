import numpy as np
import sys,h5py
from Corrfunc.mocks.DDrppi_mocks import DDrppi_mocks


#Lbox = 768
#G0 = np.loadtxt('/cosma7/data/dp004/dc-armi2/HOD_mocks/galaxy_catalogues/Galaxy_GR_z0.3_L768_HOD_13.10_14.08_13.12_0.11_1.01_pos_M200c_weight.dat')
G0_file = h5py.File('/cosma7/data/dp004/dc-armi2/HOD_mocks/lightcone/LC_GR_L768_RA_Dec_z_Mh_IDcen_z0.24_0.36_HOD_LOWZ_ld_n_wp_rp.hdf5','r')
R0_file = h5py.File('/cosma7/data/dp004/dc-armi2/HOD_mocks/lightcone/R_GR_L768_RA_Dec_z_Mh_IDcen_z0.24_0.36_HOD_LOWZ_ld_n_wp_rp.hdf5','r')

C0 = G0_file['data']
R0 = R0_file['data']

NR = len(R0)/len(C0)

npi=80
nsi=20
sigma = np.logspace(np.log10(0.5),np.log10(50.),nsi+1)
pi = np.linspace(0,100.,npi+1)
dpi = np.diff(pi)[0]
s_l = np.log10(sigma[:-1]) + np.diff(np.log10(sigma))[0]/2.
rp = 10**s_l

D1D2_est = DDrppi_mocks(autocorr=True,cosmology=2,nthreads=28,pimax=80,binfile=sigma,RA1=C0['RA'],DEC1=C0['DEC'],CZ1=C0['z'],weights1=np.ones(len(C0)),weight_type='pair_product')

D1R2_est = DDrppi_mocks(autocorr=False,cosmology=2,nthreads=28,pimax=80,binfile=sigma,RA1=C0['RA'],DEC1=C0['DEC'],CZ1=C0['comoving_distance'],weights1=np.ones(len(C0)),RA2=R0['RA'],DEC2=R0['DEC'],CZ2=R0['comoving_distance'],weights2=np.ones_like(R0['RA']),is_comoving_dist=True,weight_type='pair_product')

R1R2_est = DDrppi_mocks(autocorr=True,cosmology=2,nthreads=28,pimax=80,binfile=sigma,RA1=R0['RA'],DEC1=R0['DEC'],CZ1=R0['comoving_distance'],weights1=np.ones(len(R0)),weight_type='pair_product',is_comoving_dist=True)

D1D2 = D1D2_est['npairs']*D1D2_est['weightavg']
R1R2 = R1R2_est['npairs']*R1R2_est['weightavg'] / (NR*NR)
D1R2 = D1R2_est['npairs']*D1R2_est['weightavg'] / NR

xi2D = (D1D2 - 2*D1R2 + R1R2)/R1R2
xi_pi_sigma = np.zeros((npi,nsi))
for j in range(nsi):
    for i in range(npi):
        xi_pi_sigma[i,j] = xi2D[i+j*npi] 

wp_obs = 2*np.sum(xi_pi_sigma,axis=0)