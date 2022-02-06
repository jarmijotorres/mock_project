import healpy as hp
import numpy as np
from astropy.cosmology import Planck15 as cosmo
from scipy.spatial import Voronoi,voronoi_plot_2d,ConvexHull
from mpi4py import MPI
comm =MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def voronoi_volume(XY_box):
    v = Voronoi(XY_box[:,(0,1)])
    vol = np.zeros(v.npoints)
    for i, reg_num in enumerate(v.point_region):
        indices = v.regions[reg_num]
        if -1 in indices: # some regions can be opened
            vol[i] = np.inf
        else:
            vol[i] = ConvexHull(v.vertices[indices]).volume
    #id_cen = (v.points == np.array([0.,0.])).all(axis=1)
    #v_i = vol[id_cen][0]
    #v_mean = np.sum(vol[vol<np.inf])/len(vol[vol<np.inf])
    return vol#/v_mean

def dcomov(z):
    '''
    Include little h to obtain Mpc/h units
    '''
    return (cosmo.comoving_distance(z)*cosmo.h).value
def deg_to_Mpc_overh(z):
    '''
    Use arcmin to comoving dist at redshift z multiply by 60
    to obtain kpc per deg and divide by 1000 to get results in Mpc 
    (include little h to get Mpc/h units)
    '''
    return cosmo.h*60*(cosmo.kpc_comoving_per_arcmin(z).value)/1e3

def Mpc_overh_to_deg(z):
    '''
    Use comoving dist. per arsec at redshift z divide by 60
    to obtain kpc per arcmin (/3600 per deg.) and multiply by 1000 to get results in Mpc
    (include little h to get Mpc/h units)
    '''
    return cosmo.h*1000*(cosmo.arcsec_per_kpc_comoving(z).value)/3.6e3

def RADECZ_2_XY(G1_chunk,z_mean,alpha0):
    z = dcomov(G1_chunk[:,2])
    dc_mean = dcomov(z_mean)
    alpha = (G1_chunk[:,0])
    delta = np.radians(G1_chunk[:,1])
    y = np.sin(delta)*dc_mean
    x = (alpha - alpha0)*deg_to_Mpc_overh(z_mean)
    XY_chunk = np.array([x,y]).T
    return XY_chunk

def XY_2_alphadelta(XY_chunk,z_mean,alpha0):
    z = dcomov(z_mean)
    alpha = XY_chunk[:,0]*(Mpc_overh_to_deg(z_mean)) + alpha0
    delta = np.rad2deg(np.arcsin(XY_chunk[:,1]/z))
    
    return np.array([alpha,delta]).T
    
vol_dir = '/cosma7/data/dp004/dc-armi2/HOD_mocks/marks/slicing/'
l='/cosma7/data/dp004/dc-armi2/HOD_mocks/survey_galaxies/galaxy_LOWZ_North_z0.2_0.4.dat'
#lr='/cosma7/data/dp004/dc-armi2/HOD_mocks/survey_galaxies/random0_LOWZ_North_z0.2_0.4.dat'
lp='/cosma7/data/dp004/dc-armi2/HOD_mocks/survey_galaxies/edges_LOWZ_ran_RA_DEC_z0.2_0.4.dat'
cat_name = l.split('/')[-1]
LOWZ_ra_dec_z = np.loadtxt(l)
LOWZ_dp = np.loadtxt(lp)
#LOWZ_randoms = np.loadtxt(lr)

Nslices = 14
slice_edges = np.linspace(0.2,0.4,Nslices)

slice_i = 6
zi = slice_edges[slice_i]
zf = slice_edges[slice_i+1]
z_mean = 0.5*(zi+zf)
#ddc = dcomov(zf) - dcomov(zi)

LOWZ_slice = LOWZ_ra_dec_z[:,(0,1,2,3)][(LOWZ_ra_dec_z[:,2]>=zi)&(LOWZ_ra_dec_z[:,2]<=zf)]
wrap_slice = LOWZ_dp[(LOWZ_dp[:,2]>=zi)&(LOWZ_dp[:,2]<=zf)]

alpha0 = np.min(LOWZ_ra_dec_z[:,0])
#
XY_survey = RADECZ_2_XY(LOWZ_slice[:,(0,1,2)],z_mean,alpha0)
XY_wrap = RADECZ_2_XY(wrap_slice[:,(0,1,2)],z_mean,alpha0)
#move mask at z=0.3
#mask_dp = np.array([LOWZ_randoms[:,0],LOWZ_randoms[:,1],np.full_like(LOWZ_randoms[:,0],fill_value=z_mean)]).T
#XY_dp = RADECZ_2_XY(mask_dp[:,(0,1,2)],z_mean=z_mean,alpha0=alpha0)
#alphadelta_mask = XY_2_alphadelta(XY_dp,z_mean=z_mean,alpha0=alpha0)
#load mask
#alpha_dp = np.radians(alphadelta_mask[:,0])# from 0 to 2pi
#delta_dp = np.radians(alphadelta_mask[:,1]) + np.pi/2.;#from 0 to pi

NSIDE=1024
#p = hp.ang2pix(nside=NSIDE,phi=alpha_dp,theta=delta_dp,nest=True)

LOWZ_mask_IDs = np.loadtxt('/cosma7/data/dp004/dc-armi2/HOD_mocks/survey_galaxies/mask_surveygeometry_randoms_XY_alphadelta_LOWZ_NGC_NSIDE1024.mask')#list(set(p))
                     
ranmask = np.zeros(12*NSIDE**2)
ranmask[LOWZ_mask_IDs.astype(int)] = 1.

#np.savetxt('/cosma7/data/dp004/dc-armi2/HOD_mocks/survey_galaxies/mask_surveygeometry_randoms_XY_alphadelta_LOWZ_NGC_NSIDE1024.mask',LOWZ_mask_IDs)

G1 = np.loadtxt('/cosma7/data/dp004/dc-armi2/HOD_mocks/galaxy_catalogues/Galaxy_GR_z0.3_L1536_HOD_13.102_14.077_13.117_0.150_1.011_pos_HaloM200c_weight.dat')

Lbox = 1536#int(sys.argv[3])
Ns = 40#int(sys.argv[4])

ls = Lbox/float(Ns) #length of each slice

Zi = rank*ls
Zf = (rank+1)*ls
ddc = Zf -Zi
G1_chunk = G1[:,(0,1,2)][(G1[:,2]>=Zi)&(G1[:,2]<=Zf)]

G1_chunk_p = np.concatenate([G1_chunk + np.array([0.,-200.,0]),G1_chunk + np.array([1536.,-200.,0])],axis=0)

#simple galaxy patch + blue particles mixture
bcondx = (G1_chunk_p[:,0] >= XY_wrap[:,0].min()+10)&(G1_chunk_p[:,0] <= XY_wrap[:,0].max()-10)
bcondy = (G1_chunk_p[:,1] >= XY_wrap[:,1].min()+10)&(G1_chunk_p[:,1] <= XY_wrap[:,1].max()-10)

XY_chunk = G1_chunk_p[bcondx&bcondy]#survey masked#

alphadelta_chunk = XY_2_alphadelta(XY_chunk,z_mean=0.3,alpha0=alpha0)
alpha_chunk=np.radians(alphadelta_chunk[:,0])
delta_chunk=np.radians(alphadelta_chunk[:,1]) + np.pi/2.

is_p = hp.ang2pix(NSIDE, theta=delta_chunk, phi=alpha_chunk, nest=True)

mask_points = ranmask[is_p]==1.0
masked_chunk=XY_chunk[mask_points]

XY_gal_bp = np.concatenate([masked_chunk[:,(0,1)],XY_wrap],axis=0)#gal+bp

vol2D = voronoi_volume(XY_gal_bp)#tesselate gal+bp

vol_gals = vol2D[:len(masked_chunk)]#keep galaxies only

np.savetxt('/cosma7/data/dp004/dc-armi2/HOD_mocks/marks/slicing/V2D_slice_%d_GR_L1536_%d_slices_38.4.vol'%(rank,size),np.vstack([masked_chunk.T,vol_gals]).T)

#np.savetxt(vol_dir+'V2D_slice%d_GR_L1536_.%d_slices_%.1lf.vol'%(rank,Ns,ddc),vol_gals)

comm.barrier()

vols_all = comm.gather(vol_gals,root=0)
G1_all = comm.gather(masked_chunk,root=0)
if rank ==0:
    vols_all = np.concatenate(vols_all)
    G1_all = np.concatenate(G1_all,axis=0)
    G1_vol = np.vstack([G1_all.T,vols_all]).T
    np.savetxt(vol_dir+'V2D_GR_L1536_surveymasked.%d_slices_%.1lf.vol'%(Ns,ddc),G1_vol)
