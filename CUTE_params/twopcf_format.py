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
#Output format
xi_results = h5py.File('/cosma7/data/dp004/dc-armi2/HOD_mocks/observables/clustering/TWOPCF_rppi_mark1_CMASS_North_logdim1_0.5_50_10log10rpbins_dim2_0.5_80_50logibins_z0.46_0.54.hdf5','r')
xi2d = xi_results['full_result/xi']
rp = xi_results['full_result/Axis_1_bin_centre'][:]
drp = xi_results['full_result/Axis_1_bin_width'][:]
pi = xi_results['full_result/Axis_2_bin_centre'][:]
dpi = xi_results['full_result/Axis_2_bin_width'][:]
dlogpi = np.diff(np.log(dpi))[0]

#wp_full = np.sum(xi2d[:],axis=1)*dpi[0]
wp_full = np.sum(xi2d[:]*pi,axis=1)*dlogpi[0]
#use Jackknife info
wp_full = np.sum(xi2d[:]*pi,axis=1)*dlogpi[0]
NJK = 100
Nrp = len(rp)
Npi = len(pi)

wp_JK = []
for i in range(100):
    xi2d = xi_results['jk_reg%d/xi'%i]
    int_wp = np.sum(xi2d[:]*pi,axis=1)*dlogpi
    wp_JK.append(int_wp)
    
wp_mean = np.mean(wp_JK,axis=0)

C_ = np.zeros((Nrp,Nrp))
for i in range(len(C_)):
    for j in range(len(C_)):
        for k in range(NJK):
            C_[i][j] += (NJK-1)/float(NJK)*(wp_JK[k][i] - wp_mean[i])*(wp_JK[k][j] - wp_mean[j])
            
r_ = np.zeros_like(C_)
for i in range(len(C_)):
    for j in range(len(C_)):
        r_[i][j] = C_[i][j] / np.sqrt(C_[i][i]*C_[j][j])#normalization

f,ax = plt.subplots(1,1,figsize=(8,8))
cl = ax.imshow(r_,origin='lower',cmap='jet',extent=[np.log10(0.5),np.log10(50),np.log10(0.5),np.log10(50)])
c1 = plt.colorbar(cl,fraction=0.046, pad=0.04)
ax.set_xticks([-0.25,0.25,0.75,1.25])
ax.set_yticks([-0.25,0.25,0.75,1.25])
c1.set_label(r'$C_{ij}/\sqrt{C_{ii}C_{jj}}$')
ax.set_xlabel(r'$\log$ ($r_p$ [Mpc $h^{-1}$])')
ax.set_ylabel(r'$\log$ ($r_p$ [Mpc $h^{-1}$])')
plt.tight_layout()
plt.show()

wp_err = np.sqrt(C_.diagonal())

S = np.array([rp,wp_mean,wp_err]).T

np.savetxt('/cosma7/data/dp004/dc-armi2/HOD_mocks/observables/clustering/JK100_wp_logrp0.5_50_13logrpbins_pimax100_z0.24_0.36.txt',S)
np.savetxt('/cosma7/data/dp004/dc-armi2/HOD_mocks/observables/clustering/JK100_Cov_logrp0.5_50_13bins.txt',C_)


#============== print twopcf format =========#
#parameter file for TWOPCF
i=0
A = """data_filename = /cosma7/data/dp004/dc-armi2/HOD_mocks/galaxy_catalogues/twopcf_catalogues/V2D_8_slices_dz_0.015_Galaxy_LOWZ_North_z0.24_0.36.hdf5
data_file_type = hdf5      # ascii/hdf5
random_filename = /cosma7/data/dp004/dc-armi2/HOD_mocks/galaxy_catalogues/twopcf_catalogues/random1_LOWZ_North_z0.24_0.36.hdf5
random_file_type = hdf5    # ascii/hdf5

coord_system = equatorial            # equatorial/cartesian

ra_x_dataset_name = RA        # hdf5 dataset names
dec_y_dataset_name = DEC      # ra/dec/z for equatorial
z_z_dataset_name = Z          # x/y/z for cartesian
weight_dataset_name = mark1_shuffled{0}   # Name for weight dataset if needed
#jk_dataset_name = JK_ID

use_weights = 1    # Boolean 0/1, assumes column 4 if reading ascii file
n_threads = 0       # Set to zero for automatic thread detection

#n_jk_regions = 100

omega_m = 0.2865
h = 0.6774
z_min = 0.24
z_max = 0.36

plot_monopole = 0     # Boolean 0/1
monopole_filename = none
monopole_output_type = hdf5
monopole_log_base = 1.3 # Set to 1 for linear, any float above 1.1 valid
monopole_min = 0.0
monopole_max = 100.0
monopole_n_bins = 30

plot_sigma_pi = 1        # Boolean 0/1
sigma_pi_filename = /cosma7/data/dp004/dc-armi2/HOD_mocks/observables/clustering/TWOPCF_mark1_rppi_LOWZ_North_logdim1_0.5_50_10log10rpbins_dim2_0_80_100pibins_z0.24_0.36_shuffle{1}.dat #none
sigma_pi_output_type = hdf5 #hdf5/ascii
sigma_log_base = 1.58489319    # Set to 1 for linear, any float above 1.1 valid
sigma_min = 0.5
sigma_max = 50.0
sigma_n_bins = 10
pi_log_base = 1.0               #1.06913094     #Set to 1 for linear, any float above 1.1 valid
pi_min = 0.0
pi_max = 80.0
pi_n_bins = 100

plot_s_mu = 0        # Boolean 0/1
s_mu_filename = s_mu.hdf5
s_mu_output_type = hdf5
s_log_base = 1.3      # Set to 1 for linear, any float above 1.1 valid
s_min = 0.0
s_max = 100.0
s_n_bins = 40
mu_n_bins = 50

""".format(i,i)
with open('/cosma/home/dp004/dc-armi2/Mock_project/CUTE_params/shuffling/TWOPCF_shuffling_{}.ini'.format(i), 'w') as f:
    f.write(A)
f.close()