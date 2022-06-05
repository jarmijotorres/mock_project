from glob import glob
import numpy as np

vol_slices_GR_L1536 = glob('/cosma7/data/dp004/dc-armi2/HOD_mocks/marks/slicing/V2D_slice_*_GR_L1536_*')
vol_slices_LOWZ = glob('/cosma7/data/dp004/dc-armi2/HOD_mocks/marks/slicing/V2D_slice_*_LOWZ_*')

V1 = np.loadtxt('/cosma7/data/dp004/dc-armi2/HOD_mocks/marks/V2D_galaxy_LOWZ_North_z0.2_0.4.dat.14_slices_41.5.vol',usecols=(4,))
V0 = np.loadtxt('/cosma7/data/dp004/dc-armi2/HOD_mocks/marks/slicing/V2D_GR_L1536_surveymasked.40_slices_38.4.vol',usecols=(3,))

V_mean = 1/(2.8e-4*38.4)

V_L1536 = []
for l in vol_slices_GR_L1536:
    V_L1536.append(np.loadtxt(l))
    
V_LOWZ = []
for l in vol_slices_LOWZ:
    V_LOWZ.append(np.loadtxt(l))

for i in range(13):
    V_s = len(V_LOWZ[i])/2.8e-4
    V_b = len(V0)/2.8e-4

    N_LOWZ,_ = np.histogram(np.log10(V_LOWZ[i]/V_mean),bins=20,range=(-4,2))
    N_L1536,_ = np.histogram(np.log10(V0/V_mean),bins=20,range=(-4,2))

    n_LOWZ = N_LOWZ/V_s
    n_L1536 = N_L1536/V_b
    f,ax = plt.subplots(1,1,figsize=(6.5,6.0),sharey=True,sharex=True)
    ax.step(Vb,n_LOWZ,where='mid',linestyle='-',linewidth=2.0,color='k',label=r'LOWZ $%.2lf < z < %.2lf$'%(slice_edges[i],slice_edges[i+1]))
    ax.step(Vb,n_L1536,where='mid',linestyle='--',linewidth=2.0,color='r',label=r'Mock LOWZ')
    #ax.step(Vb,n_L1536_box,where='mid',linestyle=':',linewidth=2.0,color='r',label=r'GR $L_{box} = 1536\ h^{-1}$ Mpc')
    #ax.set_yscale('log')#
    ax.set_xlabel(r'$\log(V/\bar{V})$')
    ax.set_ylabel(r'$dn/d\log V$ [Mpc$^{-3}$ $h^3$]')
    ax.legend(loc=2,prop={'size':14})
    plt.tight_layout()
    plt.savefig('/cosma7/data/dp004/dc-armi2/HOD_mocks/plots/December2021/dn_dlogV_log_V_barV_slice_%d_survey_all_mock_logscale.pdf'%i,bbox_inches='tight')
    plt.show()
