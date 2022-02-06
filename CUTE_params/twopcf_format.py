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
xi_results = h5py.File('/cosma7/data/dp004/dc-armi2/HOD_mocks/observables/clustering/TWOPCF_rppi_LOWZ_North_logdim1_0.5_50_15log10rpbins_dim2_0.5_100_100logpibins_z0.24_0.36_JK65.dat','r')
xi2d = xi_results['full_result/xi']
rp = xi_results['full_result/Axis_1_bin_centre'][:]
drp = xi_results['full_result/Axis_1_bin_width'][:]
pi = xi_results['full_result/Axis_2_bin_centre'][:]
dpi = xi_results['full_result/Axis_2_bin_width'][:]
dlogpi = np.diff(np.log(dpi))[0]
wp_full = np.sum(xi2d[:]*pi,axis=1)*dlogpi
#use Jackknife info

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
r_ = np.zeros_like(C_)
for i in range(len(C_)):
    for j in range(len(C_)):
        for k in range(NJK):
            C_[i][j] += (NJK-1)/float(NJK)*(wp_JK[k][i] - wp_mean[i])*(wp_JK[k][j] - wp_mean[j])
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