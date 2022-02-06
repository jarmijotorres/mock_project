import numpy as np
import sys,h5py
sys.path.append('/cosma/home/dp004/dc-armi2/pywrap_HOD_fitting/src/')
from hod import *
from tpcf_obs import *
from chi2 import *

import matplotlib.pyplot as plt
import matplotlib as mpl
from cycler import cycler
import corner
plt.style.use('/cosma/home/dp004/dc-armi2/papers/presentation.mplstyle')

labs=[r"$\log\ M_{min}$", r"$\log\ M_1$", r"$\log\ M_0$", r"$\sigma$",r"$\alpha$"]

#generate HOD mock using the best fit from the MCMC run.
Mod='GR'
Lbox = 768
haloes_table = h5py.File('/cosma7/data/dp004/dc-armi2/HOD_mocks/halo_catalogues/Haloes_MG-Gadget_'+Mod+'_z0.3_L%d_ID_M200c_R200c_pos_Nsh_FirstSh_SubHaloList_SubHaloMass_logMmin_11.2.0.hdf5'%Lbox,'r')

chains = np.load('/cosma7/data/dp004/dc-armi2/mcmc_runs/outputs/chains/MCMCpost_chains_HOD_GR_L768_400it_56walkers_0.5An_0.5Awp_target_LOWZ_z0.24_0.36_err_sigma_sim_subhaloes.npy')
lklhd_chains = np.load('/cosma7/data/dp004/dc-armi2/mcmc_runs/outputs/likelihoods/MCMClklhd_chains_HOD_GR_L768_400it_56walkers_0.5An_0.5Awp_target_LOWZ_z0.24_0.36_err_sigma_sim_subhaloes.npy')
#min_lh = np.sort(lklhd_samples)[::-1]
#min_lh_arg = np.argsort(lklhd_samples)[::-1]
lklhd_samples = lklhd_chains.flatten()

samples = chains.reshape(chains.shape[0]*chains.shape[1],chains.shape[2])

prior_range = np.array([[12.5,13.5],
                        [13.0,14.6],
                        [12.5,13.8],
                        [0.0,0.6],
                        [0.7,1.2]])

ndim=5
theta_max = np.zeros(ndim)
p_bin = np.zeros((ndim,2))
for i in range(ndim):
    tm,_ = np.histogram(samples[:,i],bins=13,range=prior_range[i])
    bm = 0.5*(_[:-1] + _[1:])
    p_bin[i][0] = _[np.argmax(tm)]
    p_bin[i][1] = _[np.argmax(tm)+1]
    theta_max[i] = bm[np.argmax(tm)]
    
theta_min_lh = samples[min_lh_arg]


G0 = HOD_mock_subhaloes(theta_min_lh,haloes_table,Lbox=Lbox,weights_haloes=None)
n0 = len(G0)/(768.**3)
wp_sim0 = wp_from_box(G0,n_threads=16,Lbox = Lbox,Nsigma = 20,return_rpavg=True)

#========================== plot ===========================#
wp_data = np.loadtxt('/cosma7/data/dp004/dc-armi2/HOD_mocks/observables/clustering/JK25_wp_logrp0.5_50_20bins_pimax80_z0.24_0.36.txt')
rp = wp_data[:,0]
wp_rp = wp_data[:,1]/wp_data[:,0]
wp_err = wp_data[:,2]/wp_data[:,0]

f,ax = plt.subplots(2,1,figsize=(6,6.5),sharex=True,gridspec_kw={'height_ratios':[3,1.3]})
ax[0].errorbar(rp,wp_rp,yerr=3*wp_err,fmt='ko',label='LOWZ $0.2<z<0.4$')
for i in range(len(wp_s)):
    ax[0].plot(rp,wp_s[i],'r-')
    ax[1].plot(rp,(wp_s[i]- wp_rp)/wp_rp,'r-')
ax[1].errorbar(rp,np.zeros_like(wp_rp),yerr=3*wp_err/wp_rp,fmt='ko')
ax[1].hlines(0,0.5,50,color='grey',linewidth=2.)
ax[0].set_xscale('log')
ax[0].set_yscale('log')
ax[0].set_ylabel(r'$w_p(r_p)/r_p$')
ax[1].set_ylabel(r'Relative residual')
ax[1].set_xlabel(r'$r_p$ [Mpc $h^{-1}$]')
ax[0].legend(loc=1,prop={'size':14})
ax[1].set_ylim(-0.5,0.5)
ax[0].set_xlim(0.5,50)
plt.tight_layout()
plt.subplots_adjust(hspace=0.01)
plt.show()

np.savetxt('/cosma7/data/dp004/dc-armi2/HOD_mocks/galaxy_catalogues/boxes/Galaxy_'+Mod+'_z0.3_L'+str(Lbox)+'_HOD_%.2lf_%.2lf_%.2lf_%.2lf_%.2lf_pos_M200c_weight.dat'%tuple(theta_max),G0)


dens_kw = {'plot_density':False,'plot_datapoints':False,'data_kwargs':{'color':'k','marker':'o','ms':3,'alpha':1.0}}

fig = corner.corner(
    samples,
    labels=labs,
    bins=13,
    #fig=fig,
    #color='k',
    quantiles=None,
    range = prior_range,
    plot_contours=True,
    show_titles=False, 
    title_kwargs={"fontsize": 16},
    figsize=(7,6),
    hist_kwargs={'color':'k'},
    smooth=0.7,
    **dens_kw)

axes = np.array(fig.axes).reshape((ndim, ndim))
for i in range(ndim):
    ax = axes[i, i]
    #ax.axvline(theta_median[i], color="g")
    #ax.axvline(theta_mean[i], color="b")
    ax.axvline(theta_max[i], color="b")

for yi in range(ndim):
    for xi in range(yi):
        ax = axes[yi, xi]
        ax.plot(theta_max[xi],theta_max[yi],marker='x',color="b",ms=7,markeredgewidth=2.0)
    
plt.tight_layout()
plt.subplots_adjust(hspace=0.01,wspace=0.01)
#plt.savefig('/cosma7/data/dp004/dc-armi2/HOD_mocks/plots/January2022/MCMC_HOD.png',bbox_inches='tight')

#plot HODs
