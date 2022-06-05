import numpy as np
import h5py

in_put = '/cosma7/data/dp004/dc-armi2/HOD_mocks/galaxy_catalogues/marked_catalogues/V2D_8_slices_dz_0.015_galaxy_LOWZ_North_z0.24_0.36.hdf5'
out_dir = '/cosma7/data/dp004/dc-armi2/HOD_mocks/galaxy_catalogues/twopcf_catalogues/'

cat_name = in_put.split('/')[-1]

G0_data = h5py.File(in_put,'r')
m0 = G0_data['data']['weight']*G0_data['data']['V2D']**-0.5

hf = h5py.File(out_dir+cat_name, 'w')
hf.create_dataset('RA', data=G0_data['data']['RA'])
hf.create_dataset('DEC', data=G0_data['data']['DEC'])
hf.create_dataset('Z', data=G0_data['data']['Z'])
hf.create_dataset('weight', data=G0_data['data']['weight'])
hf.create_dataset('V2D', data=G0_data['data']['V2D'])
hf.create_dataset('mark1', data=m0)
hf.close()
#create jackknife areas
#
#
#
#

l='/cosma7/data/dp004/dc-armi2/HOD_mocks/observables/clustering/xi3D_ps_V2D_8_slices_dz_0.015_galaxy_LOWZ_North_z0.24_0.36.hdf5_logrp0.5_50_10bins_pimax80_300pibins_z0.24_0.36.txt'

xi2d_nw = np.loadtxt(l)
#xi2d_w = np.loadtxt(JK_l2[0])

#columns
#<Sigma bin number (0 to n_sigma-1)> <Pi bin number (0 to n_pi-1)> <Sigma bin centre> <Pi bin centre> <Sigma bin width> <Pi bin width> <xi> <DD> <DR> <RR>
#xi2d = np.loadtxt(l)
#### please force the 3D_ps output to be a square in shape
Nbins_sigma = 10
Nbins_pi = 300
pi_data = xi2d_nw[:Nbins_pi,0]
dpi = np.diff((pi_data))[0]
sigma_data = xi2d_nw[::Nbins_pi,1]
xi_pi_sigma = np.zeros((Nbins_pi,Nbins_sigma))
for j in range(Nbins_sigma):
    for i in range(Nbins_pi):
        xi_pi_sigma[i,j] = xi2d_nw[i+j*Nbins_pi,6]
        
wp_nw = np.zeros(Nbins_sigma)#
for i in range(Nbins_sigma):
    wp_nw[i] = 2*np.sum(xi_pi_sigma[:,i])*dpi

xi_pi_sigma = np.zeros((Nbins_pi,Nbins_sigma))
for j in range(Nbins_sigma):
    for i in range(Nbins_pi):
        xi_pi_sigma[i,j] = xi2d_w[i+j*Nbins_pi,6]
               
wp_w = np.zeros(Nbins_sigma)#2*np.sum(xi_w,axis=1)*pb*dlogpi 
for i in range(Nbins_sigma):
    wp_w[i] = 2*np.sum(xi_pi_sigma[:,i]*pi_data*dlpi)
MCF_LOWZ = (1 + wp_w/ sigma_data)/(1 + wp_nw/ sigma_data)
      
rp_i = 0.5
rp_f = 50.0
Nrpbins = 13
pimax = 100
Npibins = 300
NJK = 25
for i in range(NJK):
    A =    """data_filename= /cosma5/data/dp004/dc-armi2/SDSS/BOSS/DR12/subsamples/galaxy_CMASS_North_z0.46_0.54.dat
random_filename= /cosma5/data/dp004/dc-armi2/SDSS/BOSS/DR12/subsamples/random0_CMASS_North_z0.46_0.54.dat
input_format= 2
output_filename= /cosma7/data/dp004/dc-armi2/HOD_mocks/observables/clustering/CUTE_xi3D_CMASS_North_monopole_s_0.5_1000_20sbins.hdf5
corr_type= 3D_ps
omega_M= 0.274
omega_L= 0.726
w= -1
log_bin= 1
dim1_min_logbin= {1}
dim1_max= {2}
dim1_nbin= {3}
dim2_max= {4}
dim2_nbin= {5}
dim3_min= 0.2
dim3_max= 0.4
dim3_nbin= 1""".format(i,rp_i,rp_f,Nrpbins,pimax,Npibins)
    with open('/cosma/home/dp004/dc-armi2/Mock_project/CUTE_params/LOWZ_resampling_{}.ini'.format(i), 'w') as f:
        f.write(A)
    f.close()
    
    
    
