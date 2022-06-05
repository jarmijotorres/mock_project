import numpy as np
from uncertainties import unumpy,ufloat
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import ScalarFormatter
plt.style.use('/cosma/home/dp004/dc-armi2/papers/presentation.mplstyle')

wp_CODEX_LOWZ = np.loadtxt('/cosma7/data/dp004/dc-armi2/Jackknife_runs/JK25_wpCG_logrp0.5_50_13bins_pimax80_z0.2_0.4.txt')

rp_codex = wp_CODEX_LOWZ[:,0] 
wp_rp_codex = wp_CODEX_LOWZ[:,1] / rp_codex
wp_rp_codex_err = wp_CODEX_LOWZ[:,2] / rp_codex

Lbox=['768','1536']
L_style = ['-','--']
f,ax = plt.subplots(2,1,figsize=(6,6.5),sharex=True,gridspec_kw={'height_ratios':[3,1.3]})
ax[0].errorbar(rp_codex,wp_rp_codex,yerr=3*wp_rp_codex_err,fmt='ko',label = r'CODEX-LOWZ $0.2<z<0.4$')
ax[1].errorbar(rp,np.zeros_like(rp),yerr=3*wp_rp_codex_err/wp_rp_codex,fmt='ko')
lines = []
legend_elements = [Line2D([0], [0], color='k', lw=2,ls=L_style[0], label='Line'),
                  Line2D([0], [0], color='k', lw=2,ls=L_style[1], label='Line')]

for L,li in zip(Lbox,L_style):
    wp_CG_GR = np.loadtxt('/cosma7/data/dp004/dc-armi2/HOD_mocks/observables/clustering/JK27_wpCG_MassCut_GR_z0.3_L%s_rp_0.5_50_13rpbins.dat'%L)
    wp_CG_F5 = np.loadtxt('/cosma7/data/dp004/dc-armi2/HOD_mocks/observables/clustering/JK27_wpCG_MassCut_F5_z0.3_L%s_rp_0.5_50_13rpbins.dat'%L)

    rp = wp_CG_GR[:,0]
    wp_rp_GR = wp_CG_GR[:,1] / rp
    wp_rp_GR_err = wp_CG_GR[:,2] / rp
    #
    wp_rp_F5 = wp_CG_F5[:,1] / rp
    wp_rp_F5_err = wp_CG_F5[:,2] / rp

    wp_F5_residual = unumpy.uarray(wp_CG_F5[:,1],std_devs=wp_CG_F5[:,2])/ wp_CODEX_LOWZ[:,1]
    wp_GR_residual = unumpy.uarray(wp_CG_GR[:,1],std_devs=wp_CG_GR[:,2])/ wp_CODEX_LOWZ[:,1]

    GR_ratio = unumpy.nominal_values(wp_GR_residual)
    F5_ratio = unumpy.nominal_values(wp_F5_residual)
    #
    GR_ratio_err = unumpy.std_devs(wp_GR_residual)
    F5_ratio_err = unumpy.std_devs(wp_F5_residual)


    l1, = ax[0].plot(rp,wp_rp_GR,'r',linewidth=2.0,linestyle=li)
    l0, = ax[0].plot(rp,wp_rp_F5,'b',linewidth=2.0,linestyle=li)
    
    ax[0].fill_between(rp,np.abs(wp_rp_GR - wp_rp_GR_err),wp_rp_GR+wp_rp_GR_err,facecolor='r',alpha=0.2)
    ax[0].fill_between(rp,wp_rp_F5- wp_rp_F5_err,wp_rp_F5+wp_rp_F5_err,facecolor='b',alpha=0.2)
#
    ax[1].plot(rp,GR_ratio-1,'r-',linewidth=2.0,linestyle=li)
    ax[1].plot(rp,F5_ratio-1,'b-',linewidth=2.0,linestyle=li)
    
    ax[1].fill_between(rp,GR_ratio-GR_ratio_err-1,GR_ratio+GR_ratio_err-1,facecolor='r',alpha=0.2)
    ax[1].fill_between(rp,F5_ratio-F5_ratio_err-1,F5_ratio+F5_ratio_err-1,facecolor='b',alpha=0.2)
    lines.append([l0,l1])
c1 = ax[0].legend([lines[0][0],lines[0][1]],['GR','F5'],loc=1,bbox_to_anchor=(1,0.8),prop={'size':14})
c2 = ax[0].legend([legend_elements[0],legend_elements[1]],[r'$L_{box} = 768\ h^{-1}$ Mpc',r'$L_{box} = 1536\ h^{-1}$ Mpc'],loc=1,prop={'size':14})
ax[0].add_artist(c1)
ax[0].add_artist(c2)
ax[1].xaxis.set_major_formatter(ScalarFormatter())
ax[1].set_xticks([1,2,5,10,20,50])
ax[1].set_yticks([-1,-0.5,0,0.5,1.0])
ax[1].set_ylim(-1.0,1.0)
ax[0].set_xlim(0.5,50)
ax[0].set_ylim(1e-2,1e4)
ax[0].set_xscale('log')
ax[0].set_yscale('log')
ax[0].set_ylabel(r'$w_p(r_p)/r_p$')
ax[1].set_ylabel(r'Relative residual')
ax[1].set_xlabel(r'$r_p$ [Mpc $h^{-1}$]')
ax[0].legend(loc=4,prop={'size':14})
plt.tight_layout()
plt.subplots_adjust(hspace=0.01)
#plt.savefig('/cosma7/data/dp004/dc-armi2/HOD_mocks/plots/October2021/wp_rp_CODEX_richness_58_48-LOWZ_z0.2_0.4_MG_GADGET_L768_GR_F5_Richness_selection.pdf',bbox_inches='tight')
plt.show()