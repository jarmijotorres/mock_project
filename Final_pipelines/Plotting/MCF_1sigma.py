import numpy as np
from glob import glob
from uncertainties import unumpy,ufloat
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import ScalarFormatter
from matplotlib.lines import Line2D
plt.style.use('/cosma/home/dp004/dc-armi2/papers/presentation.mplstyle')

MCFs_GR_list = glob('/cosma7/data/dp004/dc-armi2/HOD_mocks/galaxy_catalogues/boxes/family/MCF/MCF_sspace_GR_z0.3_theta_*_mV2D20s_p-0.5.dat')
MCFs_GR_z03 = []
for l in MCFs_GR_list:
    MCFs_GR_z03.append(np.loadtxt(l,usecols=(1,)))

MCFs_GR_list = glob('/cosma7/data/dp004/dc-armi2/HOD_mocks/galaxy_catalogues/boxes/family/MCF/MCF_sspace_GR_z0.5_theta_*_mV2D20s_p-0.5.dat')
MCFs_GR_z05 = []
for l in MCFs_GR_list:
    MCFs_GR_z05.append(np.loadtxt(l,usecols=(1,)))


MCFs_F5_list = glob('/cosma7/data/dp004/dc-armi2/HOD_mocks/galaxy_catalogues/boxes/family/MCF/MCF_sspace_F5_z0.3_theta_*_mV2D20s_p-0.5.dat')
MCFs_F5_z03 = []
for l in MCFs_F5_list:
    MCFs_F5_z03.append(np.loadtxt(l,usecols=(1,)))

MCFs_F5_list = glob('/cosma7/data/dp004/dc-armi2/HOD_mocks/galaxy_catalogues/boxes/family/MCF/MCF_sspace_F5_z0.5_theta_*_mV2D20s_p-0.5.dat')
MCFs_F5_z05 = []
for l in MCFs_F5_list:
    MCFs_F5_z05.append(np.loadtxt(l,usecols=(1,)))
    
MCF_GR_z03_mean = np.median(MCFs_GR_z03,axis=0)
MCF_GR_z03_l = np.percentile(MCFs_GR_z03,q=16,axis=0)
MCF_GR_z03_h = np.percentile(MCFs_GR_z03,q=84,axis=0)

MCF_F5_z03_mean = np.median(MCFs_F5_z03,axis=0)
MCF_F5_z03_l = np.percentile(MCFs_F5_z03,q=16,axis=0)
MCF_F5_z03_h = np.percentile(MCFs_F5_z03,q=84,axis=0)

MCF_GR_z05_mean = np.median(MCFs_GR_z05,axis=0)
MCF_GR_z05_l = np.percentile(MCFs_GR_z05,q=16,axis=0)
MCF_GR_z05_h = np.percentile(MCFs_GR_z05,q=84,axis=0)

MCF_F5_z05_mean = np.median(MCFs_F5_z05,axis=0)
MCF_F5_z05_l = np.percentile(MCFs_F5_z05,q=16,axis=0)
MCF_F5_z05_h = np.percentile(MCFs_F5_z05,q=84,axis=0)

f,ax = plt.subplots(1,1,figsize=(6,6))
ax.errorbar(rp,MCF_CMASS - MCF_CMASS*0.1,yerr=MCF_err,fmt='o',color='grey')
#ax.errorbar(rp,MCF_LOWZ,yerr=MCF_err,fmt='o',color='k')
#
#ax.plot(rp,MCF_GR_z03_mean,'crimson',lw=2)
#ax.plot(rp,MCF_F5_z03_mean,'navy',lw=2)
#
ax.plot(rp,MCF_GR_z05_mean,'salmon',lw=2)
ax.plot(rp,MCF_F5_z05_mean,'deepskyblue',lw=2)

#ax.fill_between(rp,MCF_GR_z03_l,MCF_GR_z03_h,facecolor='crimson',alpha=0.5)
#ax.fill_between(rp,MCF_F5_z03_l,MCF_F5_z03_h,facecolor='navy',alpha=0.5)

ax.fill_between(rp,MCF_GR_z05_l,MCF_GR_z05_h,facecolor='salmon',alpha=0.5)
ax.fill_between(rp,MCF_F5_z05_l,MCF_F5_z05_h,facecolor='deepskyblue',alpha=0.5)

ax.set_xscale('log')
ax.set_xlabel(r'$r_p$ [Mpc $h^{-1}$]')
ax.set_ylabel(r'$\mathcal{M}(r_p)$')
ax.legend(loc=1)
#plt.savefig('/cosma7/data/dp004/dc-armi2/HOD_mocks/plots/May2022/M_rp_20_40_slices_L768.pdf',bbox_inches='tight')
plt.show()

f,ax = plt.subplots(1,1,figsize=(6,6))
#ax.errorbar(rp,MCF_CMASS,yerr=MCF_err,fmt='o',color='grey')
ax.errorbar(rp,MCF_LOWZ,yerr=MCF_err,fmt='o',color='k')
#
ax.plot(rp,MCF_GR_z03_mean,'crimson',lw=2)
ax.plot(rp,MCF_F5_z03_mean,'navy',lw=2)
#
#ax.plot(rp,MCF_GR_z05_mean,'salmon',lw=2)
#ax.plot(rp,MCF_F5_z05_mean,'deepskyblue',lw=2)

ax.fill_between(rp,MCF_GR_z03_l,MCF_GR_z03_h,facecolor='crimson',alpha=0.5)
ax.fill_between(rp,MCF_F5_z03_l,MCF_F5_z03_h,facecolor='navy',alpha=0.5)

#ax.fill_between(rp,MCF_GR_z05_l,MCF_GR_z05_h,facecolor='salmon',alpha=0.5)
#ax.fill_between(rp,MCF_F5_z05_l,MCF_F5_z05_h,facecolor='deepskyblue',alpha=0.5)

ax.set_xscale('log')
ax.set_xlabel(r'$r_p$ [Mpc $h^{-1}$]')
ax.set_ylabel(r'$\mathcal{M}(r_p)$')
ax.legend(loc=1)
#plt.savefig('/cosma7/data/dp004/dc-armi2/HOD_mocks/plots/May2022/M_rp_20_40_slices_L768.pdf',bbox_inches='tight')
plt.show()