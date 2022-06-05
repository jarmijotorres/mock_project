import numpy as np
from glob import glob
from uncertainties import unumpy,ufloat
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import ScalarFormatter
from matplotlib.lines import Line2D
plt.style.use('/cosma/home/dp004/dc-armi2/papers/presentation.mplstyle')

#wp_CODEX_LOWZ = np.loadtxt('/cosma7/data/dp004/dc-armi2/Jackknife_runs/JK25_wpCG_logrp0.5_50_13bins_pimax80_z0.2_0.4.txt')

#rp_codex = wp_CODEX_LOWZ[:,0] 
#wp_rp_codex = wp_CODEX_LOWZ[:,1] / rp_codex
#wp_rp_codex_err = wp_CODEX_LOWZ[:,2] / rp_codex

Lbox=['768','1536']
L_style = ['-','--']
legend_elements = [Line2D([0], [0], color='k', lw=2,ls=L_style[0], label='Line'),
                  Line2D([0], [0], color='k', lw=2,ls=L_style[1], label='Line')]

p = '0.25'#'%.5lf'%(1/3.)

MCF_data= np.loadtxt('/cosma7/data/dp004/dc-armi2/HOD_mocks/observables/markedclustering/JK25_MCF_mVp-'+p+'_V2D_8_slices_dz_0.015_galaxy_LOWZ_North_z0.24_0.36.hdf5_logrp0.5_50.0_10bins_pimax80_80pibins.txt')
rp = MCF_data[:,0]
MCF_LOWZ = MCF_data[:,1]

MCFs_GR_list = glob('/cosma7/data/dp004/dc-armi2/HOD_mocks/observables/markedclustering/HOD_family/MCF_GR_theta_*_mV2Dp-'+p+'.dat')
MCFs_GR = []
for l in MCFs_GR_list:
    MCFs_GR.append(np.loadtxt(l,usecols=(1,)))
MCFs_F5_list = glob('/cosma7/data/dp004/dc-armi2/HOD_mocks/observables/markedclustering/HOD_family/MCF_F5_theta_*_mV2Dp-'+p+'.dat')
MCFs_F5 = []
for l in MCFs_F5_list:
    MCFs_F5.append(np.loadtxt(l,usecols=(1,)))

MCF_GR = np.median(MCFs_GR,axis=0)
MCF_GR_err_low = np.percentile(MCFs_GR,16,axis=0)
MCF_GR_err_high = np.percentile(MCFs_GR,84,axis=0)

GR_ratio = MCF_GR/MCF_LOWZ
GR_ratio_err_low = MCF_GR_err_low / MCF_LOWZ
GR_ratio_err_high = MCF_GR_err_high / MCF_LOWZ

MCF_F5 = np.median(MCFs_F5,axis=0)
MCF_F5_err_low = np.percentile(MCFs_F5,16,axis=0)
MCF_F5_err_high = np.percentile(MCFs_F5,84,axis=0)

F5_ratio = MCF_F5/MCF_LOWZ
F5_ratio_err_low = MCF_F5_err_low / MCF_LOWZ
F5_ratio_err_high = MCF_F5_err_high / MCF_LOWZ
#MCF_F5 = np.median(MCFs_F5,axis=0)
#MCF_F5_err = np.std(MCFs_F5,axis=0)
#MCF_F5_residual = unumpy.uarray(MCF_F5,std_devs=MCF_F5_err)/ MCF_LOWZ
#F5_ratio = MCF_F5/MCF_LOWZ
#F5_ratio_err = unumpy.std_devs(MCF_F5_residual)


f,ax = plt.subplots(2,1,figsize=(6,6.5),sharex=True,gridspec_kw={'height_ratios':[3,1.3]})
#for i in range(len(MCFs_GR)):
l0, = ax[0].plot(rp,MCF_GR_median,'r',linewidth=2.0,linestyle='-')
#l1, = ax[0].plot(rp,np.mean(MCFs_F5,axis=0),'b',linewidth=2.0,linestyle='-')

ax[0].fill_between(rp,MCF_GR_err_low,MCF_GR_err_high,facecolor='r',alpha=0.2)
#ax[0].fill_between(rp,MCF_F5- MCF_F5_err,MCF_F5+MCF_F5_err,facecolor='b',alpha=0.2)
#
ax[1].plot(rp,GR_ratio-1,'red',linewidth=2.0,linestyle='-')
#ax[1].plot(rp,F5_ratio-1,'b-',linewidth=2.0,linestyle='-')

ax[1].fill_between(rp, GR_ratio_err_low -1,GR_ratio_err_high-1,facecolor='r',alpha=0.2)
#ax[1].fill_between(rp,F5_ratio-F5_ratio_err-1,F5_ratio+F5_ratio_err-1,facecolor='b',alpha=0.2)
    
ax[0].errorbar(rp,MCF_LOWZ,yerr=MCF_data[:,2],fmt='ko',label=r'LOWZ $0.24<z<0.36$')
ax[1].errorbar(rp,np.zeros_like(MCF_LOWZ),yerr=MCF_data[:,2]/MCF_LOWZ,fmt='ko')
c1 = ax[0].legend([l0],['GR'],loc=1,bbox_to_anchor=(1,0.99),prop={'size':14})
ax[0].add_artist(c1)
ax[1].xaxis.set_major_formatter(ScalarFormatter())
ax[1].set_xticks([1,2,5,10,20,50])
ax[1].set_yticks([-1.0,-0.5,0.0,0.5,1.0,1.5,2.0])
ax[1].set_ylim(-0.5,0.5)
ax[0].set_xlim(0.5,50)
#ax[0].set_ylim(0.9,9)
#ax[0].set_ylim(1e-2,1e4)
ax[0].set_xscale('log')
#ax[0].set_yscale('log')
ax[0].set_ylabel(r'$\mathcal{M}(r_p)$')
ax[1].set_ylabel(r'Relative residual')
ax[1].set_xlabel(r'$r_p$ [Mpc $h^{-1}$]')
ax[0].legend(loc=1,prop={'size':14},bbox_to_anchor=(1.0,0.8))
plt.tight_layout()
plt.subplots_adjust(hspace=0.01)
#plt.savefig('/cosma7/data/dp004/dc-armi2/HOD_mocks/plots/March2022/M_rp_LOWZ_z0.24_0.36_MG_L768_GR_F5_mVp-'+p+'.pdf',bbox_inches='tight')
plt.show()