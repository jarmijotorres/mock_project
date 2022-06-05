import numpy as np
import sys,h5py
from scipy.special import erf
from scipy.interpolate import interp1d
sys.path.append('/cosma/home/dp004/dc-armi2/codes/py_codes/')
from binning_data import binning_XY

theta_GR = np.array([13.102, 14.0771384 , 13.11738439,  0.15,  1.011])
theta_F5 = np.array([13.16076923, 13.982307692, 13.580, 0.11153846, 0.99153846])  
#logMmin, logM1, logM0, sigma, alpha = theta

#
def HOD_analytic(M,theta):
    logMmin, logM1, logM0, sigma, alpha = theta
    Ncen = 0.5*(1.0+erf((np.log10(M)-logMmin)/sigma))
    Nsat = np.zeros_like(Ncen)
    bM = M > 10**logM0
    Nsat[bM] = Ncen[bM]*((M[bM]-(10**logM0))/(10**logM1))**alpha
    return Ncen+Nsat

def Ncen_fraction(haloes,theta,Mbins):
    logMmin, logM1, logM0, sigma, alpha = theta
    Ncen = 0.5*(1.0+erf((np.log10(haloes['M200c'])-logMmin)/sigma))
    r1 = np.random.random(size=len(Ncen))
    is_cen = np.zeros_like(Ncen,dtype=int)
    b2 = Ncen >=r1
    is_cen[b2] = 1
    Ncen_bins,_ = binning_XY(np.log10(haloes['M200c']),is_cen,BINS=Mbins)
    return Ncen_bins[:,2]

def Nsat_fraction(haloes,theta,Mbins):
    logMmin, logM1, logM0, sigma, alpha = theta
    Ncen = 0.5*(1.0+erf((np.log10(haloes['M200c'])-logMmin)/sigma))
    Nsat = np.zeros_like(Ncen)
    b1 = haloes['M200c'] > 10**logM0
    Nsat[b1] = Ncen[b1]*((haloes['M200c'][b1]-(10**logM0))/(10**logM1))**alpha
    r2 = np.random.poisson(lam=Nsat,size=len(Nsat))
    #
    Nsats_in_haloes = r2[haloes['M200c']>10**(logM0)]
    haloes_with_sats = haloes[haloes['M200c']>10**(logM0)]
    Nsh_in_haloes = haloes['Nsh'][haloes['M200c']>10**(logM0)]
    Haloes_Nsat_Nsh = np.zeros((len(haloes_with_sats),2))
    for i in range(len(Haloes_Nsat_Nsh)):
        Haloes_Nsat_Nsh[i,0] = Nsats_in_haloes[i]
        Haloes_Nsat_Nsh[i,1] = Nsh_in_haloes[i]
    Nsat_final = np.min(Haloes_Nsat_Nsh[:,(0,1)],axis=1)
    Nsat_bins,_ = binning_XY(np.log10(haloes_with_sats['M200c']),Nsat_final,BINS=Mbins)
    N_sats_table = Nsat_bins[:,2]
    N_sats_table[np.isnan(N_sats_table)] = 0
    return N_sats_table
    

N_GR = HOD_analytic(M=10**Mrange,theta=theta_GR)
N_F5 = HOD_analytic(M=10**Mrange,theta=theta_F5)

Lbox = 768.
haloes_table = h5py.File('/cosma7/data/dp004/dc-armi2/HOD_mocks/halo_catalogues/Haloes_MG-Gadget_GR_z0.3_L%d_ID_M200c_R200c_pos_Nsh_FirstSh_SubHaloList_SubHaloMass_logMmin_11.2.0.hdf5'%Lbox,'r')
MainHaloes_GR = haloes_table['MainHaloes']
SubHaloes_GR = haloes_table['SubHaloes']

haloes_table = h5py.File('/cosma7/data/dp004/dc-armi2/HOD_mocks/halo_catalogues/Haloes_MG-Gadget_F5_z0.3_L%d_ID_M200c_R200c_pos_Nsh_FirstSh_SubHaloList_SubHaloMass_logMmin_11.2.0.hdf5'%Lbox,'r')
MainHaloes_F5 = haloes_table['MainHaloes']
SubHaloes_F5 = haloes_table['SubHaloes']

Mbins = np.arange(12.0,15.5,0.1)
N_cen_GR = Ncen_fraction(haloes=MainHaloes_GR,theta=theta_GR,Mbins=Mbins)
N_sat_GR = Nsat_fraction(haloes=MainHaloes_GR,theta=theta_GR,Mbins=Mbins)

N_cen_F5 = Ncen_fraction(haloes=MainHaloes_F5,theta=theta_F5,Mbins=Mbins)
N_sat_F5 = Nsat_fraction(haloes=MainHaloes_F5,theta=theta_F5,Mbins=Mbins)

f,ax = plt.subplots(figsize=(6.5,6))
l1, = ax.plot(10**Mrange,N_GR[0],'r-',linewidth=2.0,)
ax.plot(10**Mrange,N_GR[1],'r--',linewidth=2.0)
ax.scatter(10**N_cen_GR[:,1],N_cen_GR[:,2],c='r',s=50)
ax.scatter(10**N_sat_GR[:,1],N_sat_GR[:,2],c='r',s=50,marker='s')
#ax.plot(10**Mrange,N_F5[0]+N_F5[1],'b-',linewidth=2.0,label=r'F5')
l2, = ax.plot(10**Mrange,N_F5[0],'b-',linewidth=2.0,)
ax.plot(10**Mrange,N_F5[1],'b--',linewidth=2.0,)
ax.scatter(10**N_cen_F5[:,1],N_cen_F5[:,2],c='b',s=50)
ax.scatter(10**N_sat_F5[:,1],N_sat_F5[:,2],c='b',s=50,marker='s')
l3 = ax.vlines(0,0,1,linewidth=2.0,color='k')
l4 = ax.vlines(0,0,1,linewidth=2.0,color='k',linestyle='--')
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlim(5e12,1e15)
ax.set_ylim(1e-2,1e1)
ax.set_xlabel(r'$M_{200c}\ [M_{\odot}\ h^{-1}]$')
ax.set_ylabel(r'$<N_{gal}>$')
c1 = ax.legend([l1,l2],['GR','F5'],loc=2,prop={'size':14})
c2 = ax.legend([l3,l4],['Central','satellites'],loc=4,prop={'size':14})
ax.add_artist(c1)
ax.add_artist(c2)
plt.tight_layout()
plt.savefig('/cosma7/data/dp004/dc-armi2/HOD_mocks/plots/October2021/HOD_GR_F5.pdf',bbox_inches='tight')
plt.show()


#Add HOD*HMF 
HMF_box_GR = np.loadtxt('/cosma7/data/dp004/dc-armi2/HMF_weights/halo_massFunction/HMF_MG-Gadget_GR_z0.3_L%d_dlogM0.15.dat'%Lbox)
HMF_box_F5 = np.loadtxt('/cosma7/data/dp004/dc-armi2/HMF_weights/halo_massFunction/HMF_MG-Gadget_F5_z0.3_L%d_dlogM0.15.dat'%Lbox)

hmf_interp_GR = interp1d(np.log10(HMF_box_GR[:,2]),np.log10(HMF_box_GR[:,3]),kind='linear')
hmf_interp_F5 = interp1d(np.log10(HMF_box_F5[:,2]),np.log10(HMF_box_F5[:,3]),kind='linear')

Mrange = np.arange(12.5,np.log10(HMF_box_GR[-1,2]),0.01)

HMF_HOD_prod_GR = (10**hmf_interp_GR(Mrange))*HOD_analytic(M=10**Mrange,theta=theta_GR)
HMF_HOD_prod_F5 = (10**hmf_interp_F5(Mrange))*HOD_analytic(M=10**Mrange,theta=theta_F5)

Mbins = np.arange(12.5,np.log10(HMF_box_GR[-1,2]),0.1)
HMF_HOD_bins_GR = (10**hmf_interp_GR(Mb))*(N_cen_GR+N_sat_GR)
HMF_HOD_bins_F5 = (10**hmf_interp_F5(Mb))*(N_cen_F5+N_sat_F5)

f,ax = plt.subplots(figsize=(6.5,6))
l1, = ax.plot(10**Mrange,(HMF_HOD_prod_GR[0]+HMF_HOD_prod_GR[1])/np.sum(HMF_HOD_prod_GR[0]+HMF_HOD_prod_GR[1]),'r-',linewidth=2.0,)
ax.plot(10**Mb,HMF_HOD_bins_GR/np.sum(HMF_HOD_prod_GR[0]+HMF_HOD_prod_GR[1]),'rs')
ax.plot(10**Mb,HMF_HOD_bins_F5/np.sum(HMF_HOD_prod_F5[0]+HMF_HOD_prod_F5[1]),'bs')
l2, = ax.plot(10**Mrange,(HMF_HOD_prod_F5[0]+HMF_HOD_prod_F5[1])/np.sum(HMF_HOD_prod_F5[0]+HMF_HOD_prod_F5[1]),'b-',linewidth=2.0,)
#l3 = ax.vlines(0,0,1,linewidth=2.0,color='k')
#l4 = ax.vlines(0,0,1,linewidth=2.0,color='k',linestyle='--')
ax.set_xscale('log')
#ax.set_yscale('log')
ax.set_xlim(5e12,1e15)
ax.set_ylim(0.0,0.0165)
ax.set_xlabel(r'$M_{200c}\ [M_{\odot}\ h^{-1}]$')
ax.set_ylabel(r'$n_{gal}(M)$')
c1 = ax.legend([l1,l2],['GR','F5'],loc=1,prop={'size':14})
#c2 = ax.legend([l3,l4],['Central','satellites'],loc=4,prop={'size':14})
ax.add_artist(c1)
#ax.add_artist(c2)
plt.tight_layout()
plt.savefig('/cosma7/data/dp004/dc-armi2/HOD_mocks/plots/November2021/HOD_HMF_GR_F5_L1536.pdf',bbox_inches='tight')
plt.show()


f,ax = plt.subplots(1,1,figsize=(6.5,6))
for alpha in high_alpha:
    theta = np.array([13.1,high_logM1,logM0,0.3,alpha])
    Ncen,Nsat = HOD_analytic(10**Mrange,theta=theta)
    ax.plot(10**Mrange,Ncen+Nsat,'-')
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_ylim(1e-2,1e2)
ax.set_xlim(3e12,3e15)
ax.text(0.05,0.9,r'$\log M_0 = %.2lf$'%logM0,transform=ax.transAxes)
ax.text(0.05,0.85,r'$\log M_1 = \log 4M_0$',transform=ax.transAxes)
ax.text(0.05,0.80,r'$%.1lf < \alpha < %.1lf$'%(high_alpha[0],high_alpha[-1]),transform=ax.transAxes)
ax.set_xlabel(r'$M_{200c}\ [M_{\odot}\ h^{-1}]$')
ax.set_ylabel(r'$<N>$')
#ax.set_xlabel("number of samples, $N$")
#ax.set_ylabel(r"$\tau$ estimates")
#plt.legend(fontsize=14);
plt.tight_layout()
plt.savefig('/cosma7/data/dp004/dc-armi2/HOD_mocks/plots/March2022/HOD_high_alpha_logM1_log5M0_13.30.pdf',bbox_inches='tight')
plt.show()