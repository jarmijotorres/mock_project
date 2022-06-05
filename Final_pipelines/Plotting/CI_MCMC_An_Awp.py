import numpy as np
import h5py
from scipy.special import erf
from scipy.interpolate import interp1d
from scipy.stats import norm

import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import ScalarFormatter
plt.style.use('/cosma/home/dp004/dc-armi2/papers/presentation.mplstyle')

def HOD_analytic(M,theta):
    logMmin, logM1, logM0, sigma, alpha = theta
    Ncen = 0.5*(1.0+erf((np.log10(M)-logMmin)/sigma))
    Nsat = np.zeros_like(Ncen)
    bM = M > 10**logM0
    Nsat[bM] = Ncen[bM]*((M[bM]-(10**logM0))/(10**logM1))**alpha
    return Ncen+Nsat

wp_LOWZ = np.loadtxt('/cosma7/data/dp004/dc-armi2/HOD_mocks/observables/clustering/JK100_LOWZ_North_wp_logrp0.5_50_13logrpbins_pimax80_z0.24_0.36.txt')

rp = wp_LOWZ[:,0] 
wp_rp = wp_LOWZ[:,1] / rp
wp_rp_err = wp_LOWZ[:,2] / rp
rpbins = np.logspace(np.log10(0.5),np.log10(0.5),len(rp)+1)

An=0.5
Awp = 0.5

dc_mock = h5py.File('/cosma7/data/dp004/dc-armi2/HOD_mocks/galaxy_catalogues/boxes/family/datacube/GR_'+str(An)+'An_'+str(Awp)+'Awp_chi_2_1sigma_ConfInt_1000_rs_theta_wp_rp_n_gal.hdf5','r')

wps_mock = dc_mock['wp_rp'][:]
ns_mock = dc_mock['n_gal'][:]
theta_mock = dc_mock['theta'][:]

n_std = np.std(ns_mock)

wp_GR_l = np.min(wps_mock_GR,axis=0)
wp_GR_h = np.max(wps_mock_GR,axis=0)

wp_F5_l = np.min(wps_mock_F5,axis=0)
wp_F5_h = np.max(wps_mock_F5,axis=0)

###clustering
f,ax = plt.subplots(2,1,figsize=(6,6.5),sharex=True,gridspec_kw={'height_ratios':[3,1.3]})

l1 = ax[0].fill_between(rp,wp_GR_l,wp_GR_h,facecolor='r',alpha=0.5)
l2 = ax[0].fill_between(rp,wp_F5_l,wp_F5_h,facecolor='b',alpha=0.5)

ax[1].fill_between(rp,wp_GR_l/wp_rp - 1,wp_GR_h/wp_rp - 1,facecolor='r',alpha=0.5)
ax[1].fill_between(rp,wp_F5_l/wp_rp - 1,wp_F5_h/wp_rp - 1,facecolor='b',alpha=0.5)
    
ax[0].errorbar(rp,wp_rp,yerr=3*wp_rp_err,fmt='ko',label = r'LOWZ $0.24<z<0.36$',zorder=10)
ax[1].errorbar(rp,np.zeros_like(rp),yerr=3*wp_rp_err/wp_rp,fmt='ko',zorder=11)   
ax[0].set_xscale('log')
ax[0].set_yscale('log')
ax[1].xaxis.set_major_formatter(ScalarFormatter())
ax[1].set_xticks([1,2,5,10,20,50])
ax[1].set_yticks([-1,-0.5,0,0.5,1.0])
ax[1].set_ylim(-1.0,1.0)
ax[0].set_xlim(0.5,50)
ax[0].set_ylim(1e-2,1e3)
ax[0].set_xscale('log')
ax[0].set_yscale('log')
ax[0].set_ylabel(r'$w_p(r_p)/r_p$')
ax[1].set_ylabel(r'Relative residual')
ax[1].set_xlabel(r'$r_p$ [Mpc $h^{-1}$]')
c1 = ax[0].legend([l1,l2],['GR','F5'],prop={'size':14})
c2 = ax[0].legend(loc=3,prop={'size':14})
ax[0].add_artist(c1)
ax[0].add_artist(c2)
plt.tight_layout()
plt.subplots_adjust(hspace=0.01)
plt.savefig('/cosma7/data/dp004/dc-armi2/HOD_mocks/plots/May2022/wp_rp_GR_F5_1sigma_chi2_mock_'+str(An)+'An_'+str(Awp)+'_scatter.pdf',bbox_inches='tight')
plt.show()
###

#number density
mean, var, skew, kurt = norm.stats(moments='mvsk')
x = np.linspace(norm.ppf(0.0001,scale=(n_std*1e4)),
                norm.ppf(0.9999,scale=(n_std*1e4)), 100)


n_target = 2.8e-4
N_mock,_ = np.histogram((ns_mock-n_target)*1e4,bins=20,range=(-5*(n_std*1e4),5*(n_std*1e4)),density=True)
nb = 0.5*(_[:-1]+_[1:])
f,ax = plt.subplots(1,1,figsize=(6.5,6))
ax.step(nb,N_mock,where='mid',color='r',linewidth=2.0,label='GR')
ax.vlines(0,0,200,ls='--',lw=2,color='k')
ax.plot(x, norm.pdf(x,scale=(n_std*1e4)),'k', lw=5, alpha=0.6, label='norm pdf')
#ax.fill_between([2.8-0.05*2.8,2.8+0.05*2.8],[0,100],facecolor='grey',alpha=0.3)
ax.set_ylim(0,2)
ax.set_xlabel(r'$n_{gal}\ \times 10^4$ [Mpc$^{-3}$ $h^3$]')
plt.savefig('/cosma7/data/dp004/dc-armi2/HOD_mocks/plots/May2022/ngal_dist_F5_1sigma_chi2_mock_'+str(An)+'An_'+str(Awp)+'wp.pdf',bbox_inches='tight')
plt.show()
###


#HODs
Mrange = np.arange(12.3,15.3,0.01)
f,ax = plt.subplots(1,1,figsize=(6.5,6))
for t in theta_mock:
    Ngals = HOD_analytic(10**Mrange,theta=t)
    ax.plot(10**Mrange,Ngals,'r-',alpha=0.2)
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_ylim(1e-2,1e2)
ax.set_xlim(3e12,2e15)
ax.set_xlabel(r'$M_{200c}\ [M_{\odot}\ h^{-1}]$')
ax.set_ylabel(r'$<N>$')
#ax.set_xlabel("number of samples, $N$")
#ax.set_ylabel(r"$\tau$ estimates")
#plt.legend(fontsize=14);
plt.tight_layout()
plt.savefig('/cosma7/data/dp004/dc-armi2/HOD_mocks/plots/May2022/HOD_1sigma_chi2_mock_'+str(An)+'An_'+str(Awp)+'Awp.pdf',bbox_inches='tight')
plt.show()
###

#wp_bins 
wp_data_lowlim = wp_rp - wp_rp_err
wp_data_highlim = wp_rp + wp_rp_err
for i in range(len(dc_mock)):
N_mock,_ = np.histogram(wps[:,i],bins=20,range=(wps[:,i].min(),wps[:,i].max()),density=True)
nb = 0.5*(_[:-1]+_[1:])
f,ax = plt.subplots(1,1,figsize=(6.5,6))
ax.step(nb,N_mock,where='mid',color='r',linewidth=2.0,label='GR')
ax.vlines(wp_rp[i],0,200,ls='--',lw=2,color='k')
#ax.fill_between([2.8-0.05*2.8,2.8+0.05*2.8],[0,100],facecolor='grey',alpha=0.3)
ax.set_ylim(0,2)
#ax.set_xlabel(r'$n_{gal}\ \times 10^4$ [Mpc$^{-3}$ $h^3$]')
#plt.savefig('/cosma7/data/dp004/dc-armi2/HOD_mocks/plots/May2022/ngal_dist_1sigma_chi2_mock_'+str(An)+'An_'+str(Awp)+'wp.pdf',bbox_inches='tight')
plt.show()