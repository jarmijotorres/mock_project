import numpy as np
import h5py
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import ScalarFormatter
from matplotlib.lines import Line2D
plt.style.use('/cosma/home/dp004/dc-armi2/papers/presentation.mplstyle')
from astropy.cosmology import Planck15 as cosmo

def dcomov(z):
    '''
    Include little h to obtain Mpc/h units
    '''
    return (cosmo.comoving_distance(z)*cosmo.h).value

G0 = h5py.File('/cosma7/data/dp004/dc-armi2/HOD_mocks/lightcone/LC_GR_L768_RA_Dec_z_Mh_IDcen_z0.24_0.36_HOD_LOWZ_hd_n_wp_rp.hdf5','r')
R0 = h5py.File('/cosma7/data/dp004/dc-armi2/HOD_mocks/lightcone/R_GR_L768_RA_Dec_z_Mh_IDcen_z0.24_0.36_HOD_LOWZ_hd_n_wp_rp.hdf5','r')

RLC = R0['data']
LC = G0['data']

LC_data = np.loadtxt('/cosma7/data/dp004/dc-armi2/HOD_mocks/survey_galaxies/galaxy_LOWZ_North_z0.24_0.36.dat')
RLC_data = np.loadtxt('/cosma7/data/dp004/dc-armi2/HOD_mocks/survey_galaxies/random0_LOWZ_North_z0.24_0.36.dat')

Z = LC['z']
Zd = LC_data[:,2]

LOWZ_area_deg = 5836.21
LOWZ_area_rad = LOWZ_area_deg*(np.pi/180.)**2
Nslices=8
slice_edges = np.linspace(0.24,0.36,Nslices+1)
dV_dz_s = LOWZ_area_rad/3 * (dcomov(slice_edges[1:])**3 - dcomov(slice_edges[:-1])**3)#com. volume
zb = 0.5*(slice_edges[:-1] + slice_edges[1:])

new_LC = []
new_RLC = []
for i in range(Nslices):
    LC_slice = LC[(Z>slice_edges[i])&(Z<slice_edges[i+1])]
    RLC_slice = RLC[(RLC['comoving_distance']>dcomov(slice_edges[i]))&(RLC['comoving_distance']<dcomov(slice_edges[i+1]))]
    LCd_slice = Zd[(Zd>slice_edges[i])&(Zd<slice_edges[i+1])]
    RLCd_slice = RLC_data[(RLC_data[:,2]>slice_edges[i])&(RLC_data[:,2]<slice_edges[i+1])]
    
    N_zd = len(LCd_slice)#/dV_dz_s[i])
    Nr_zd = len(RLCd_slice)
    
    np.random.shuffle(LC_slice)
    np.random.shuffle(RLC_slice)
    
    rs_slice = LC_slice[:N_zd]
    Rrs_slice = RLC_slice[:Nr_zd]
    new_LC.append(rs_slice)
    new_RLC.append(Rrs_slice)
    
new_LC = np.concatenate(new_LC)
new_RLC = np.concatenate(new_RLC)

n_z = np.zeros(Nslices)
n_zd = np.zeros(Nslices)
nr_z = np.zeros(Nslices)
nr_zd = np.zeros(Nslices)
for i in range(Nslices):
    LC_slice = new_LC[(new_LC['z']>slice_edges[i])&(new_LC['z']<slice_edges[i+1])]
    LCd_slice = Zd[(Zd>slice_edges[i])&(Zd<slice_edges[i+1])]
    RLC_slice = new_RLC[(new_RLC['comoving_distance']>dcomov(slice_edges[i]))&(new_RLC['comoving_distance']<dcomov(slice_edges[i+1]))]
    RLCd_slice = RLC_data[(RLC_data[:,2]>slice_edges[i])&(RLC_data[:,2]<slice_edges[i+1])]
    
    n_z[i] = len(LC_slice)/dV_dz_s[i]
    nr_z[i] = len(RLC_slice)/dV_dz_s[i]
    n_zd[i] = len(LCd_slice)/dV_dz_s[i]
    nr_zd[i] = len(RLCd_slice)/dV_dz_s[i]
    

c = ['k','r','b']
l = [r'LOWZ $0.24<z<0.36$',r'Mock LC low density',r'Mock LC high density']
L_style = ['-','--']
legend_elements = [Line2D([0], [0], color='k', lw=2,ls=L_style[0], label='Line'),
                  Line2D([0], [0], color='k', lw=2,ls=L_style[1], label='Line')]

f,ax = plt.subplots(1,1,figsize=(6.5,6))
for i in range(3):
    l1, = ax.step(zb,n_Z[i]*1e4,where='mid',linestyle='-',linewidth=2.0,color=c[i],label=l[i])
    ax.step(zb,(n_rZ[i]/50)*1e4,where='mid',linestyle='--',linewidth=2.0,color=c[i])
ax.set_xlim(zb[0],zb[-1])
ax.set_ylim(1,5)
ax.set_xlabel(r'$z$')
ax.set_ylabel(r'$10^4\times n(z)$ [Mpc$^{-3}\ h^3$]')
c1 = ax.legend(loc=2,prop={'size':14})
c2 = ax.legend([legend_elements[0],legend_elements[1]],['Galaxies','Randoms'],loc=3,prop={'size':14})
ax.add_artist(c1)
ax.add_artist(c2)
plt.tight_layout()