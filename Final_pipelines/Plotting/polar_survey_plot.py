import numpy as np
import h5py
import sys,time
sys.path.append('/cosma/home/dp004/dc-armi2/pywrap_HOD_fitting/src/')
from hod import *
from tpcf_obs import *
from chi2 import *
from survey_geometry import *

import matplotlib.pyplot as plt
plt.style.use('/cosma/home/dp004/dc-armi2/papers/presentation.mplstyle')

G0 = h5py.File('/cosma7/data/dp004/dc-armi2/HOD_mocks/galaxy_catalogues/twopcf_catalogues/V2D_8_slices_dz_0.015_galaxy_LOWZ_North_z0.24_0.36.hdf5')

Ns=16
slice_edges2 = np.linspace(0.24,0.36,Ns+1)

cp = 10
dec_slice = (G0['DEC'][:] > cp - 2.5)&(G0['DEC'][:] < cp + 2.5)
ra_rad = np.radians(G0['RA'][:][dec_slice])
z = G0['Z'][:][dec_slice]
fig, ax = plt.subplots(figsize=(9,9),subplot_kw={'projection': 'polar'})
ax.plot(ra_rad - np.pi/2,z,'k.',ms=0.4)
#for zi in slice_edges:
#    ax.plot(tcirc,np.full_like(tcirc,zi),'b--',linewidth=1.0,alpha=0.5)
for zi in slice_edges2:
    ax.plot(tcirc,np.full_like(tcirc,zi),'r--',linewidth=1.0,alpha=0.5)

ax.set_rorigin(0.1)
ax.set_rlim(0.24,0.36)
ax.set_thetalim(np.pi/4,3*np.pi/4)
ax.grid(False)
plt.tight_layout()
plt.savefig('/cosma7/data/dp004/dc-armi2/HOD_mocks/plots/May2022/LOWZ_slices2_polarplot.pdf',bbox_inches='tight')
plt.show()

Ns=32
for i in range(Ns):
    slice_edges = np.linspace(0.24,0.36,Ns+1)
    cp = slice_edges[i]
    z_slice = (G0['Z'][:] > slice_edges[i])&(G0['Z'][:] < slice_edges[i+1])
    RA = np.radians(G0['RA'][:][z_slice])
    DEC = G0['DEC'][:][z_slice]

    wz_slice = (survey_dp[:,2] > slice_edges[i])&(survey_dp[:,2] < slice_edges[i+1])
    wRA = np.radians(survey_dp[:,0][wz_slice])
    wDEC = survey_dp[:,1][wz_slice]

    f,ax = plt.subplots(1,1,figsize=(10,6))
    ax.plot(RA,np.sin(np.radians(DEC)),'k.',ms=0.5)
    ax.plot(wRA,np.sin(np.radians(wDEC)),'b.',ms=2.5)
    ax.set_xlim(wRA.min(),wRA.max())
    ax.set_ylim(np.sin(np.radians(wDEC.min())),np.sin(np.radians(wDEC.max())))
    ax.set_xlabel(r'$\alpha$ [radians]')
    ax.set_ylabel(r'$\sin{\delta}$')
    plt.tight_layout()
    plt.savefig('/cosma7/data/dp004/dc-armi2/HOD_mocks/plots/May2022/LOWZ_RADEC_slice_z%lf_%lf.png'%(slice_edges[i],slice_edges[i+1]),bbox_inches='tight')
    plt.show()