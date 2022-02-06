import numpy as np
import sys,h5py
from mpi4py import MPI
from astropy.cosmology import Planck15 as cosmo
from astropy.table import Table
from scipy.spatial import Voronoi,ConvexHull

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
    return vol

def dcomov(z):
    '''
    include little h to obtain Mpc/h units
    '''
    return (cosmo.comoving_distance(z)*cosmo.h).value
def deg_to_Mpc_overh(z):
    '''
    use arcmin to comoving at redshift z divide by 0.25 
    to obtain kpc per deg and divide by 1000 to get results in Mpc 
    (include little h to get Mpc/h units)
    '''
    return cosmo.h*60*(cosmo.kpc_comoving_per_arcmin(z).value)/1e3

def RADECZ_2_XY(G1_chunk,z_mean,alpha0):
    z = dcomov(G1_chunk[:,2])
    dc_mean = dcomov(z_mean)
    alpha = (G1_chunk[:,0])
    delta = np.radians(G1_chunk[:,1])
    y = np.sin(delta)*dc_mean
    x = (alpha - alpha0)*deg_to_Mpc_overh(z_mean)
    XY_chunk = np.array([x,y]).T
    return XY_chunk

vol_dir = '/cosma7/data/dp004/dc-armi2/HOD_mocks/galaxy_catalogues/marked_catalogues/'
#l='/cosma7/data/dp004/dc-armi2/HOD_mocks/survey_galaxies/galaxy_LOWZ_North_z0.2_0.4.dat'
l=sys.argv[1]
cat_name = l.split('/')[-1]
cat_data = h5py.File(l,'r')
G0 = np.array(cat_data['data'])
alpha0 = np.min(G0['RA'])
survey_dp = np.loadtxt('/cosma7/data/dp004/dc-armi2/HOD_mocks/galaxy_catalogues/survey_catalogues/edges_LOWZ_ran_RA_DEC_z0.2_0.4.dat')

slice_edges = np.linspace(0.24,0.36,size+1)
zi = slice_edges[rank]
zf = slice_edges[rank+1]
ddc = dcomov(zf) - dcomov(zi)
z_mean = 0.5*(zi+zf)
dz = zf-zi

G0_slice = G0[(G0['Z']>=zi)&(G0['Z']<=zf)]

wrap_slice = survey_dp[(survey_dp[:,2]>=zi)&(survey_dp[:,2]<=zf)]

fill_slice = np.array([G0_slice['RA'],G0_slice['DEC'],G0_slice['Z']]).T

XY_fill = RADECZ_2_XY(fill_slice,z_mean,alpha0)
XY_wrap = RADECZ_2_XY(wrap_slice,z_mean,alpha0)

XY_chunk = np.concatenate([XY_fill[:,(0,1)],XY_wrap[:,(0,1)]],axis=0)

V2D = voronoi_volume(XY_chunk)
vol2D_chunk = V2D[:G0_slice.shape[0]]

comm.barrier()
vols_all = comm.gather(vol2D_chunk,root=0)
G1_all = comm.gather(G0_slice,root=0)
if rank ==0:
    vols = np.concatenate(vols_all)
    G1 = np.concatenate(G1_all,axis=0)
    
    G1_vol = Table(data=[G1['RA'],G1['DEC'],G1['Z'],G1['weight'],vols],names=['RA','DEC','Z','weight','V2D'])
    #np.array([G1['RA'],G1['DEC'],G1['Z'],G1['weight']*vols]).T
    #np.savetxt(vol_dir+'V2D_%d_slices_dz_%.3lf_'%(size,dz)+cat_name+'.dat',G1_vol)
    G1_vol.write(vol_dir+'V2D_%d_slices_dz_%.3lf_'%(size,dz)+cat_name,format='hdf5',path='data')
