import numpy as np
import healpy as hp
from scipy.spatial import cKDTree
from astropy.coordinates import SkyCoord
import h5py,sys
from astropy.cosmology import Planck15 as cosmo
from astropy.table import Table
#from astropy.cosmology import z_at_value
from scipy.optimize import root

def dcomov(z):
    '''
    Include little h to obtain Mpc/h units
    '''
    return (cosmo.comoving_distance(z)*cosmo.h).value

def dc2z(z,dc):
    return dcomov(z) - dc

NSIDE=1024
LOWZ_mask_IDs = np.loadtxt('/cosma5/data/dp004/dc-armi2/SDSS/BOSS/DR12/mask/randoms_LOWZ_North_NS1024.mask')
ranmask = np.zeros(12*NSIDE**2)
ranmask[LOWZ_mask_IDs.astype(int)] = 1.

G1 = np.loadtxt('/cosma7/data/dp004/dc-armi2/HOD_mocks/galaxy_catalogues/boxes/Galaxy_GR_z0.3_L768_HOD_13.07_13.90_13.33_0.15_1.01_pos_M200c_weight.dat')
G1_BB = []
for i in [-1,0,1]:
    for j in [-1,0,1]:
        for k in [-1,0,1]:
            G1_BB.append(G1[:,(0,1,2,3,5)]+np.array([i,j,k,0,0])*768.0)
G1_BB = np.concatenate(G1_BB,axis=0)

p0 = np.concatenate((np.mean(G1[:,(0,1,2)],axis=0),[0.0,0.0]))
observer_frame = G1_BB - p0
del G1_BB

nn_tree = cKDTree(observer_frame[:,(0,1,2)])

zi = 0.24
zf = 0.36
origin = np.array([0.0,0.0,0.0])
ri = dcomov(zi)
rf = dcomov(zf)
#lc_z02 = nn_tree.query_ball_point(x=origin,r=ri,)
lc_z04 = nn_tree.query_ball_point(x=origin,r=rf,)

#LC_M = observer_frame[lc_z02]
LC_L = observer_frame[lc_z04]

c = SkyCoord(x=LC_L[:,0], y=LC_L[:,1], z=LC_L[:,2], unit='Mpc', representation_type='cartesian')
c.representation_type = 'spherical'


#phi = c.ra.value[(c.distance.value > 100)&(c.distance.value < 150)]
#theta = c.dec.value[(c.distance.value > 100)&(c.distance.value < 150)]

RA = c.ra.value#to(u.radian).value#[(c.dec.value > 0)&(c.dec.value < 5)]
DEC = c.dec.value#
r = c.distance.value#[(c.dec.value > 0)&(c.dec.value < 5)]
#z = comoving2z(r)

Halo_mass = LC_L[:,3]
ID_sat = LC_L[:,4]

cat = Table(data = [RA,DEC,r,Halo_mass,ID_sat],names=['RA','DEC','comoving_distance','HaloMass','IsCen'])

cat_rs = cat[cat['comoving_distance']>ri]

#apply mask

alpha = np.radians(cat_rs['RA'])
delta = np.radians(cat_rs['DEC']) + np.pi/2.

is_p = hp.ang2pix(NSIDE, theta=delta, phi=alpha, nest=True)

mask_points = ranmask[is_p]==1.0

masked_LC = cat_rs[mask_points]

z = np.zeros(len(masked_LC))
for i in range(len(z)):
    sol = root(dc2z,x0=0.2,args=masked_LC['comoving_distance'][i],tol=1e-5,)
    z[i] = sol.x[0]

masked_LC['Z'] = z
masked_LC['weight'] = np.ones(len(z))

masked_LC.write('/cosma7/data/dp004/dc-armi2/HOD_mocks/galaxy_catalogues/lightcones/LC_GR_L768_RA_Dec_z_Mh_IDcen_weight_z0.24_0.36_HOD_LOWZ_md_n_wp_rp.hdf5',format='hdf5',path='data')

#f, ax = plt.subplots(subplot_kw={'projection': 'polar'},figsize=(10.,10.))
#ax.plot(phi, r,'k.',ms=1.0)
#ax.set_rlim(200,500)
#ax.set_rorigin(-200)
#plt.show()

#repeat for randoms 
G1 = 768.0*np.random.uniform(size=(len(G1),3))

G1_BB = []
for i in [-1,0,1]:
    for j in [-1,0,1]:
        for k in [-1,0,1]:
            G1_BB.append(G1[:,(0,1,2)]+np.array([i,j,k])*768.0)
G1_BB = np.concatenate(G1_BB,axis=0)

p0 = np.mean(G1[:,(0,1,2)],axis=0)
observer_frame = G1_BB - p0
del G1_BB

nn_tree = cKDTree(observer_frame[:,(0,1,2)])

zi = 0.24
zf = 0.36
origin = np.array([0.0,0.0,0.0])
ri = dcomov(zi)
rf = dcomov(zf)
#lc_z02 = nn_tree.query_ball_point(x=origin,r=ri,)
lc_z04 = nn_tree.query_ball_point(x=origin,r=rf,)

#LC_M = observer_frame[lc_z02]
LC_L = observer_frame[lc_z04]

c = SkyCoord(x=LC_L[:,0], y=LC_L[:,1], z=LC_L[:,2], unit='Mpc', representation_type='cartesian')
c.representation_type = 'spherical'


#phi = c.ra.value[(c.distance.value > 100)&(c.distance.value < 150)]
#theta = c.dec.value[(c.distance.value > 100)&(c.distance.value < 150)]

RA = c.ra.value#to(u.radian).value#[(c.dec.value > 0)&(c.dec.value < 5)]
DEC = c.dec.value#
r = c.distance.value#[(c.dec.value > 0)&(c.dec.value < 5)]
#z = comoving2z(r)

cat_r = Table(data = [RA,DEC,r],names=['RA','DEC','comoving_distance'])

cat_rr = cat_r[cat_r['comoving_distance']>ri]

cat_rr = np.array(cat_rr)

z = np.zeros(len(cat_rr))
for i in range(len(z)):
    sol = root(dc2z,x0=0.3,args=cat_rr['comoving_distance'][i],tol=1e-5,)
    z[i] = sol.x[0]

cat_rr = Table(cat_rr)
cat_rr['Z'] = z

rand_runs = []
for i in range(50):
    rand_ra = cat_rr['RA'][np.random.randint(low=0,high=len(cat_rr),size=len(cat_rr))]
    rand_dec = cat_rr['DEC'][np.random.randint(low=0,high=len(cat_rr),size=len(cat_rr))]
    rand_z = cat_rr['Z'][np.random.randint(low=0,high=len(cat_rr),size=len(cat_rr))]
    S = np.array([rand_ra,rand_dec,rand_z]).T
    rand_runs.append(S)
rand_runs = np.concatenate(rand_runs)

is_p = hp.ang2pix(NSIDE, theta=np.radians(rand_runs[:,1])+np.pi/2., phi=np.radians(rand_runs[:,0]), nest=True)
mask_points = ranmask[is_p]==1.0
masked_R = rand_runs[mask_points]

ran_cat = Table(data=[masked_R[:,0],masked_R[:,1],dcomov(masked_R[:,2]),masked_R[:,2],np.ones_like(masked_R[:,2])],names=['RA','DEC','comoving_distance','Z','weight'])

ran_cat.write('/cosma7/data/dp004/dc-armi2/HOD_mocks/galaxy_catalogues/lightcones/R_LC_F5_L768_RA_Dec_z_Mh_IDcen_z0.24_0.36_HOD_LOWZ_md_n_wp_rp.hdf5',format='hdf5',path='data')