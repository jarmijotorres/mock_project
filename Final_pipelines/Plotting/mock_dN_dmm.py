import numpy as np
from glob import glob
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import ScalarFormatter
from matplotlib.lines import Line2D
plt.style.use('/cosma/home/dp004/dc-armi2/papers/presentation.mplstyle')

MCFs_GR_list = glob('/cosma7/data/dp004/dc-armi2/HOD_mocks/galaxy_catalogues/boxes/family/marks/histograms/dN_dlogmm_GR_*_sep_r_0.5_50_10logrbins.dat')
dn_GR = []
for l in MCFs_GR_list:
    dn_GR.append(np.loadtxt(l))
dn_F5 = []
MCFs_F5_list = glob('/cosma7/data/dp004/dc-armi2/HOD_mocks/galaxy_catalogues/boxes/family/marks/histograms/dN_dlogmm_F5_*_sep_r_0.5_50_10logrbins.dat')
for l in MCFs_F5_list:
    dn_F5.append(np.loadtxt(l))
    
mmb = dn_GR[0][:,0]

dn_F5 = np.array(dn_F5)
dn_GR = np.array(dn_GR)

dn_dmm_GR = dn_GR[:,:,1:]
dn_dmm_F5 = dn_F5[:,:,1:]

dn_dmm_F5_med = np.median(dn_dmm_F5,axis=0)
dn_dmm_GR_med = np.median(dn_dmm_GR,axis=0)

#dn_dmm_F5_low = np.percentile(dn_dmm_F5,,axis=0)
#dn_dmm_GR_low = np.percentile(dn_dmm_GR,,axis=0)

for i in range(10):
    f,ax = plt.subplots(1,1,figsize=(6.5,6))
    ax.step(mmb,dn_dmm_GR_med[:,i],where='pre',linewidth=2.0,color='r')
    ax.step(mmb,dn_dmm_F5_med[:,i],where='pre',linewidth=2.0,color='b')
    ax.set_xlabel(r'$\log (m_im_j/\bar{m}^2)$')
    ax.set_ylabel(r'$dn/d\log m_im_j$')
    ax.text(0.3,0.90,r'$%.2lf < r /$ [Mpc $h^{-1}$] $< %.2lf$'%(sigma[i],sigma[i+1]),transform=ax.transAxes)
    ax.set_xlim(-1.0,3.5)
    ax.set_ylim(0.0,0.3)
    plt.tight_layout()
    plt.savefig('/cosma7/data/dp004/dc-armi2/HOD_mocks/plots/April2022/N_mm_sep_GR_F5_%i_100HODs.pdf'%i,bbox_inches='tight')
plt.show()