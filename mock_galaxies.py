import numpy as np
import sys,h5py
sys.path.append('/cosma/home/dp004/dc-armi2/pywrap_HOD_fitting/src/')
from hod import *
from Corrfunc.theory import xi

Lbox = 1536.

#=========== HOD mock catalogue ===========#

#haloes = np.load('/cosma7/data/dp004/dc-armi2/HOD_mocks/halo_catalogues/Haloes_MG-Gadget_GR_z0.3_L1536_ID_M200c_R200c_pos_Nsh_SubHaloList_logMmin_11.2.0.npy',allow_pickle=True)
ws = h5py.File('/cosma7/data/dp004/dc-armi2/HMF_weights/weights/weights_haloes_MG-Gadget_GR_z0.3_L1536_ID_M200c_weight_logMhalomin_11.2.hdf5','r')
w_data = ws['data']
w1 = w_data['weight']


#satellites problem
theta = np.array([12.6, 14.0 , 12.8,  0.1,  1.0]) 
logMmin, logM1, logM0, sigma, alpha = theta
Mbins = np.arange(logM0,15.0,0.1)
dlogM=0.01
Mrange = np.arange(logM0+dlogM,np.log10(haloes['M200c'].max()),dlogM)
#============ satellite estimation ===========#
Ncen = 0.5*(1.0+erf((np.log10(haloes['M200c'])-logMmin)/sigma))
r1 = np.random.random(size=len(Ncen))#
Nsat = np.zeros_like(Ncen)
b1 = haloes['M200c'] > 10**logM0#boolean 1: haloes that may contain satellites depending on M0
Nsat[b1] = Ncen[b1]*((haloes['M200c'][b1]-(10**logM0))/(10**logM1))**alpha# function for number of satellites
r2 = np.random.poisson(lam=Nsat,size=len(Nsat))

haloes_with_sats = haloes[haloes['M200c']>10**(logM0)]
#haloes_with_subhaloes = haloes['Nsub'][sh_table['M200c']>10**(logM0)]
Nsats_in_haloes = r2[haloes['M200c']>10**(logM0)]
Nsh_in_haloes = haloes['Nsh'][haloes['M200c']>10**(logM0)]

Haloes_Nsat_Nsh = np.zeros((len(haloes_with_sats),2))
for i in range(len(Haloes_Nsat_Nsh)):
    Haloes_Nsat_Nsh[i,0] = Nsats_in_haloes[i]
    Haloes_Nsat_Nsh[i,1] = Nsh_in_haloes[i]
    
    
Nsat_final = np.min(Haloes_Nsat_Nsh[:,(0,1)],axis=1)
Nsat_mean_incomplete, _ = binning_XY(np.log10(haloes_with_sats['M200c']),Nsat_final,BINS=Mbins)
Nsat_mean_complete, _ = binning_XY(np.log10(haloes_with_sats['M200c']),Nsats_in_haloes,BINS=Mbins)

Ncen_analytic = 0.5*(1.0+erf((Mrange-logMmin)/sigma))
Nsat_analytic = Ncen_analytic*((10**Mrange-(10**logM0))/(10**logM1))**alpha
Nsat_interp = interp1d(Mrange,np.log10(Nsat_analytic),kind='linear')
Nsat_bins = 10**Nsat_interp(Nsat_mean_incomplete[:,1])

#================== plot ===================#
f,ax = plt.subplots(2,1,figsize=(6,6.5),sharex=True,gridspec_kw={'height_ratios':[3,1.3]})
ax[0].plot(10**Mrange,Nsat_analytic,'k--',label=r'$N_{sat}$')
ax[0].plot(10**Nsat_mean_incomplete[:,1],Nsat_mean_incomplete[:,2],'g*',label=r'$L_{box} = 1536\ h^{-1}$ Mpc')
ax[0].set_xlim(3e12,3e15)
ax[0].set_ylim(1e-4,20)
ax[0].set_ylabel(r'$<N>$')
ax[0].set_yscale('log')
ax[0].set_xscale('log')
ax[0].legend(loc=2,prop={'size':14})
#
ax[1].plot(10**Mrange,np.ones_like(Mrange),color='k',linestyle='--')
ax[1].plot(10**Nsat_mean_incomplete[:,1],Nsat_mean_incomplete[:,2]/ Nsat_bins,'g*')
ax[1].set_ylim(0,1.2)
ax[1].set_ylabel('ratio')
ax[1].set_xlabel(r'$\log M_{200c} [M_{\odot}\ h^{-1}]$')
plt.tight_layout()
plt.subplots_adjust(hspace=0.01)
#plt.savefig('/cosma7/data/dp004/dc-armi2/HOD_mocks/plots/September2021/HOD_%.1lf_%.1lf_%.1lf_%.1lf_%.1lf_Nsatdeficit.pdf'%(logMmin,logM0,logM1,sigma,alpha),bbox_inches='tight')
plt.show()
#====================================================#


#correct satellite portions for incomplete low resolution haloes
theta = np.array([12.8, 14.0 , 13.0,  0.3,  1.0])
logMmin, logM1, logM0, sigma, alpha = theta #unpack parameters
Ncen = 0.5*(1.0+erf((np.log10(haloes['M200c'])-logMmin)/sigma))#function for Number of centrals
w1 = np.ones_like(haloes['M200c'])
r1 = np.random.random(size=len(Ncen))#MC random to populate haloes with centrals
Nsat = np.zeros_like(Ncen)
b1 = haloes['M200c'] > 10**logM0#boolean 1: haloes that may contain satellites depending on M0
Nsat[b1] = Ncen[b1]*((haloes['M200c'][b1]-(10**logM0))/(10**logM1))**alpha# function for number of satellites
r2 = np.random.poisson(lam=Nsat,size=len(Nsat))# Poisson random for the number of satellites
is_cen = np.zeros_like(Ncen,dtype=int)# flag: halo in catalogue has central
b2 = Ncen >=r1#boolean 2: MH step to populate halo with central
is_cen[b2] = 1
b3 = (Ncen < r1)&(r2>0)#boolean 3: halo has no central but will produce satellites (in this case 1 satellite becomes the central)
is_cen[b3] = 1
r2[b3] = r2[b3] - 1# for those haloes the number of satellites decrease by 1
b4 = r2 > 0# boolean 4: halo with central has satellites
has_sat = np.zeros_like(Ncen,dtype=int)#flag: halo has satellites
has_sat[b4] = 1
#
halo_with_sat = haloes[b4]# new variable to iterate over haloes with satellites only
weights_hsat = w1[b4]
Nsat_perhalo = r2[b4]

missing_sat = 0
Mass_missing_sat = []
remaining_sats=[]
Mass_remaining_sat=[]
mock = []#list to gather the catalogue
for i,hi in enumerate(halo_with_sat):
    pos_sh = hi['SubHaloList']
    np.random.shuffle(pos_sh)
    Nsubh = len(pos_sh)
    Nsat_hi =  Nsat_perhalo[i]
    Nfin = np.min((Nsubh,Nsat_hi))
    Nvar = Nsat_hi - Nsubh
    if Nvar>0:
        missing_sat += Nvar
        Mass_missing_sat.append(int(Nvar)*[np.log10(hi['M200c'])])
    elif Nvar<0:
        remaining_sats.append(pos_sh[Nfin:])
        Mass_remaining_sat.append(int(-Nvar)*[np.log10(hi['M200c'])])
    pos_satellites = pos_sh[:Nfin]
    xyz_censat = np.vstack([hi['pos'],pos_satellites])
    mass = np.full(len(xyz_censat),hi['M200c'])
    w_halo = np.full(len(xyz_censat),1.0)#hi[6] is the weight
    id_cen_sat = np.full_like(mass,-1)
    id_cen_sat[0] = 1
    halo_censat = np.vstack([xyz_censat.T,mass,w_halo,id_cen_sat]).T# all haloes with satellites
    mock.append(halo_censat)#
#
Mass_missing_sat = np.concatenate(Mass_missing_sat)
Remaining_sat = np.concatenate(remaining_sats)
Mass_remaining_sat = np.concatenate(Mass_remaining_sat)
Mbins=np.arange(logM0,np.max(Mass_missing_sat),0.1)
Mbin_sat_missing = np.digitize(Mass_missing_sat,Mbins)
Mbin_sat_remaining = np.digitize(Mass_remaining_sat,Mbins)

Mbins = np.append(Mbins,np.max(Mass_remaining_sat))


haloes_with_sats = haloes[haloes['M200c']>10**(logM0)]
#haloes_with_subhaloes = haloes['Nsub'][sh_table['M200c']>10**(logM0)]
Nsats_in_haloes = r2[haloes['M200c']>10**(logM0)]
Nsh_in_haloes = haloes['Nsh'][haloes['M200c']>10**(logM0)]

Haloes_Nsat_Nsh = np.zeros((len(haloes_with_sats),2))
for i in range(len(Haloes_Nsat_Nsh)):
    Haloes_Nsat_Nsh[i,0] = Nsats_in_haloes[i]
    Haloes_Nsat_Nsh[i,1] = Nsh_in_haloes[i]
Nsat_final = np.min(Haloes_Nsat_Nsh[:,(0,1)],axis=1)
Nsat_mean_incomplete, _ = binning_XY(np.log10(haloes_with_sats['M200c']),Nsat_final,BINS=Mbins)
Nsat_mean_complete, _ = binning_XY(np.log10(haloes_with_sats['M200c']),Nsats_in_haloes,BINS=Mbins)
dlogM=0.01
Mrange = np.arange(logM0+dlogM,np.log10(haloes['M200c'].max()),dlogM)
Ncen_analytic = 0.5*(1.0+erf((Mrange-logMmin)/sigma))
Nsat_analytic = Ncen_analytic*((10**Mrange-(10**logM0))/(10**logM1))**alpha
Nsat_interp = interp1d(Mrange,np.log10(Nsat_analytic),kind='linear')
Nsat_bins = 10**Nsat_interp(Nsat_mean_incomplete[:,1])

N_counted_before = Nsat_mean_incomplete[:,4]*Nsat_mean_incomplete[:,2]
N_expected = Nsat_mean_complete[:,4]*Nsat_mean_complete[:,2]
N_missing = N_expected - N_counted_before

recovered_sat = []
recovered_sat_mass = []
N_recovered_per_bin = []
missing_sat_per_Mbin = 0
for c in range(len(Mbins)-1):
    #Ml = Mbins[c]
    #Mh = Mbins[c+1]
    #M_pick = Mbin_sat_missing[Mbin_sat_missing == c+1] 
    M_pool = Mbin_sat_remaining[Mbin_sat_remaining == c+1]
    
    N_pool = len(M_pool) 
    N_to_pick = int(N_missing[c])#len(M_pick) + missing_sat_per_Mbin
    
    N_borr = np.min((N_pool,N_to_pick))
    
    if N_pool - N_to_pick < 0:
        missing_sat_per_Mbin = N_to_pick - N_pool
    else:
        missing_sat_per_Mbin = 0
        np.random.shuffle(Remaining_sat[Mbin_sat_remaining == c+1])
    N_recovered_per_bin.append(N_borr)
    sats_borrowed = Remaining_sat[Mbin_sat_remaining == c+1][:N_borr]
    Mass_sats_borrowed = Mass_remaining_sat[Mbin_sat_remaining == c+1][:N_borr]
    recovered_sat.append(sats_borrowed)
    recovered_sat_mass.append(Mass_sats_borrowed)
recovered_sat = np.concatenate(recovered_sat)
recovered_sat_mass = np.concatenate(recovered_sat_mass)
#

N_counted_after = np.array(N_recovered_per_bin) + N_counted_before 
N_sat_recovered = N_counted_after / Nsat_mean_incomplete[:,4]

mock_rec = np.vstack([recovered_sat.T,10**recovered_sat_mass,np.ones_like(recovered_sat_mass),np.full_like(recovered_sat_mass,-1)]).T
    
b5 = (is_cen == 1)&(~b4)#boolean 5: all haloes frome above - haloes with satellites
New_haloes = haloes[b5]
haloes_cen = np.vstack([New_haloes['pos'][:,0],New_haloes['pos'][:,1],New_haloes['pos'][:,2],New_haloes['M200c'],w1[b5],np.full(len(w1[b5]),1.0)]).T
mock.append(haloes_cen)
mock_cat = np.concatenate(mock)# array containing the catalogue

mock_final = np.vstack([mock_cat,mock_rec])
#above verbatim copy of my code, it generates HOD cen and sats, but a portion of sats is missing due to box resolution.
G0 = mock_cat
G1 = np.vstack([mock_cat,mock_rec])

ri=0.5
rf=50.0
Nb=25
rbins = np.logspace(np.log10(ri),np.log10(rf),Nb)

xi_L1536_r = xi(boxsize=1536,nthreads=4,binfile=rbins,X=G1[:,0],Y=G1[:,1],Z=G1[:,2],weights=G1[:,4],weight_type='pair_product',output_ravg=True,)

xi_L1536 = xi(boxsize=1536,nthreads=4,binfile=rbins,X=G0[:,0],Y=G0[:,1],Z=G0[:,2],weights=G0[:,4],weight_type='pair_product',output_ravg=True,)

np.savetxt('/cosma7/data/dp004/dc-armi2/HMF_weights/xi_r/xi_galaxies/xi_gg_z0.3_r_0.5_50.0_25_rbins_logMmin%.2lf_logM0%.2lf_logM1%.2lf_sigma%.2lf_alpha%.2lf_GR_L1536_r.dat'%tuple(theta),np.array([xi_L1536_r['ravg'],xi_L1536_r['xi']]).T)
np.savetxt('/cosma7/data/dp004/dc-armi2/HMF_weights/xi_r/xi_galaxies/xi_gg_z0.3_r_0.5_50.0_25_rbins_logMmin%.2lf_logM0%.2lf_logM1%.2lf_sigma%.2lf_alpha%.2lf_GR_L1536.dat'%tuple(theta),np.array([xi_L1536['ravg'],xi_L1536['xi']]).T)


theta = np.array([12.6, 14.0 , 12.8,  0.1,  1.0])
xi_L1536_r = np.loadtxt('/cosma7/data/dp004/dc-armi2/HMF_weights/xi_r/xi_galaxies/xi_gg_z0.3_r_0.5_50.0_25_rbins_logMmin%.2lf_logM0%.2lf_logM1%.2lf_sigma%.2lf_alpha%.2lf_GR_L1536_r.dat'%tuple(theta))
xi_L768_r = np.loadtxt('/cosma7/data/dp004/dc-armi2/HMF_weights/xi_r/xi_galaxies/xi_gg_z0.3_r_0.5_50.0_25_rbins_logMmin%.2lf_logM0%.2lf_logM1%.2lf_sigma%.2lf_alpha%.2lf_GR_L768_r.dat'%tuple(theta))

xi_L1536 = np.loadtxt('/cosma7/data/dp004/dc-armi2/HMF_weights/xi_r/xi_galaxies/xi_gg_z0.3_r_0.5_50.0_25_rbins_logMmin%.2lf_logM0%.2lf_logM1%.2lf_sigma%.2lf_alpha%.2lf_GR_L1536.dat'%tuple(theta))
xi_L768 = np.loadtxt('/cosma7/data/dp004/dc-armi2/HMF_weights/xi_r/xi_galaxies/xi_gg_z0.3_r_0.5_50.0_25_rbins_logMmin%.2lf_logM0%.2lf_logM1%.2lf_sigma%.2lf_alpha%.2lf_GR_L768.dat'%tuple(theta))

f,ax = plt.subplots(2,1,figsize=(6,6.5),sharex=True,gridspec_kw={'height_ratios':[3,1.3]})
ax[0].plot(xi_L768[:,0],xi_L768[:,1],'r-',label=r'$L_{box}=768\ h^{-1}$ Mpc')
ax[0].plot(xi_L768_r[:,0],xi_L768_r[:,1],'r--')
ax[0].plot(xi_L1536[:,0],xi_L1536[:,1],'g-',label=r'$L_{box}=1536\ h^{-1}$ Mpc')
ax[0].plot(xi_L1536_r[:,0],xi_L1536_r[:,1],'g--')
#ax[0].plot(rb,xi_L1536_r['xi'],'g--',label=r'weighted')
ax[0].set_xscale('log')
ax[0].set_yscale('log')
ax[1].xaxis.set_major_formatter(ScalarFormatter())
ax[1].plot(xi_L1536[:,0],0.03+(xi_L1536[:,1]-xi_L768[:,1])/xi_L768[:,1],'g-')
ax[1].plot(xi_L1536_r[:,0],0.03+(xi_L1536_r[:,1]-xi_L768[:,1])/xi_L768[:,1],'g--')
ax[1].plot(xi_L768[:,0],np.zeros_like(xi_L1536[:,0]),'r-')
ax[1].set_xticks([1,2,5,10,20,50])
ax[1].set_ylim(-0.5,0.5)
ax[0].set_xlim(0.5,50)
ax[0].set_ylim(5e-3,2e2)
ax[0].set_ylabel(r'$\xi(r)$')
ax[1].set_ylabel(r'Relative residual')
ax[1].set_xlabel(r'$r$ [Mpc $h^{-1}$]')
ax[0].legend(loc=1,prop={'size':14})
plt.tight_layout()
plt.subplots_adjust(hspace=0.01)
plt.savefig('/cosma7/data/dp004/dc-armi2/HOD_mocks/plots/September2021/2PCF_MG_GADGET_L768_1536_weigthed_logMmin_%.2lf_%.2lf_%.2lf_sigma%.1lf_%.1lf.pdf'%tuple(theta),bbox_inches='tight')
plt.show()