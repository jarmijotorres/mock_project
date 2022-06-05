import numpy as np
import healpy as hp
from copy import deepcopy
import h5py
from scipy.spatial import Voronoi,ConvexHull

def dcomov(z):
    '''
    Include little h to obtain Mpc/h units
    '''
    return (cosmo.comoving_distance(z)*cosmo.h).value

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


NSIDE=1024
rmsk = '/cosma5/data/dp004/dc-armi2/SDSS/BOSS/DR12/mask/survey_wrap_NGC_0.2_0.4_NS1024.mask'
rana = np.loadtxt(rmsk).astype(int) #pixels filled by the randoms
ranmask = np.zeros(12*NSIDE**2)
ranmask[rana.astype(int)] = 1.
ramin = np.min(hp.pix2ang(NSIDE,rana,nest=True,lonlat=False)[1])
ramax = np.max(hp.pix2ang(NSIDE,rana,nest=True,lonlat=False)[1])
decmin = np.min(hp.pix2ang(NSIDE,rana,nest=True,lonlat=False)[0])
decmax = np.max(hp.pix2ang(NSIDE,rana,nest=True,lonlat=False)[0])

nd=1.0
ranRADec=[]
for i in range(int(10*nd*len(rana))):
    rara = np.random.uniform(ramin,ramax)
    radec = np.random.uniform(decmin,decmax)
    p=hp.ang2pix(NSIDE,theta=radec,phi=rara,nest=True)
    if ranmask[p] == 1.0:
        ranRADec.append([rara, radec])
ranRADec = np.array(ranRADec)

cat = h5py.File('/cosma7/data/dp004/dc-armi2/HOD_mocks/galaxy_catalogues/marked_catalogues/V2D_8_slices_dz_0.015_galaxy_LOWZ_North_z0.24_0.36.hdf5','r')
G0 = cat['data']
alpha0 = np.min(G0['RA'])
raz = G0['Z'][np.random.randint(0,high=len(G0['Z']),size=len(ranRADec))]#boostrap random redshift
#
ranRADecZ = np.array([np.rad2deg(ranRADec.T[0]),np.rad2deg(ranRADec.T[1])-90,raz]).T

ranRADecZ = np.load('/cosma7/data/dp004/dc-armi2/HOD_mocks/galaxy_catalogues/survey_catalogues/edges_45timesNgal_LOWZ_ran_RA_DEC_z0.24_0.36.dat.npy')

IDs_rand = np.arange(0,len(ranRADecZ))
randoms_IDs = []
Nr = np.linspace(len(G0['Z']),10*len(G0['Z']),10,dtype=int)
for i in range(10):
    np.random.shuffle(IDs_rand)
    patch_nr = deepcopy(IDs_rand)[:Nr[i]]
    randoms_IDs.append(patch_nr)

size = 8
vols_G0 = []
vols_R0 = []
for ID_r in randoms_IDs:
    survey_dp = ranRADecZ[ID_r]
    slice_edges = np.linspace(0.24,0.36,size+1)
    V0 = []
    RV0 = []
    for rank in range(size):
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
        vol2D_wrap = V2D[G0_slice.shape[0]:]
        V0.append(vol2D_chunk)
        RV0.append(vol2D_wrap)
    V0 = np.concatenate(V0)
    RV0 = np.concatenate(RV0)
    vols_G0.append(V0)
    vols_R0.append(RV0)
    
    
def auto_marked_correlation_function(C0,R0,V0,sigma):
    
    dpi = 1.0#np.diff(pi)[0]
    s_l = np.log10(sigma[:-1]) + np.diff(np.log10(sigma))[0]/2.
    rp = 10**s_l
    NR = 50.0#ratio between the lenght of R0 and C0

    m0 = C0['weight']*V0**-0.5
    m0_bar = np.mean(m0)

    D1D2_est = DDrppi_mocks(autocorr=True,cosmology=2,nthreads=16,pimax=pimax,binfile=sigma,RA1=C0['RA'],DEC1=C0['DEC'],CZ1=C0['Z'],weights1=C0['weight'],weight_type='pair_product')

    D1R2_est = DDrppi_mocks(autocorr=False,cosmology=2,nthreads=16,pimax=pimax,binfile=sigma,RA1=C0['RA'],DEC1=C0['DEC'],CZ1=C0['Z'],weights1=C0['weight'],RA2=R0['RA'],DEC2=R0['DEC'],CZ2=R0['Z'],weights2=R0['weight'],weight_type='pair_product')

    R1R2_est = DDrppi_mocks(autocorr=True,cosmology=2,nthreads=16,pimax=pimax,binfile=sigma,RA1=R0['RA'],DEC1=R0['DEC'],CZ1=R0['Z'],weights1=R0['weight'],weight_type='pair_product')

    D1D2 = D1D2_est['npairs']*D1D2_est['weightavg']
    R1R2 = R1R2_est['npairs']*R1R2_est['weightavg'] / (NR*NR)
    D1R2 = D1R2_est['npairs']*R1R2_est['weightavg'] / NR

    xi2D = (D1D2 - D1R2 - D1R2 + R1R2)/R1R2
    xi_pi_sigma = np.zeros((npi,nsi))
    for j in range(nsi):
        for i in range(npi):
            xi_pi_sigma[i,j] = xi2D[i+j*npi] 

    wp_unmarked = 2*np.sum(xi_pi_sigma,axis=0)*dpi

    W1W2_est = DDrppi_mocks(autocorr=True,cosmology=2,nthreads=16,pimax=pimax,binfile=sigma,RA1=C0['RA'],DEC1=C0['DEC'],CZ1=C0['Z'],weights1=m0,weight_type='pair_product')

    W1R2_est = DDrppi_mocks(autocorr=False,cosmology=2,nthreads=16,pimax=pimax,binfile=sigma,RA1=C0['RA'],DEC1=C0['DEC'],CZ1=C0['Z'],weights1=m0,RA2=R0['RA'],DEC2=R0['DEC'],CZ2=R0['Z'],weights2=np.ones_like(R0['RA']),weight_type='pair_product')

    W1W2 = W1W2_est['npairs']*W1W2_est['weightavg']
    W1R2 = W1R2_est['npairs']*W1R2_est['weightavg']*m0_bar / NR
    R1R2 = R1R2_est['npairs']*m0_bar*m0_bar / (NR*NR)

    xi2D = (W1W2 - W1R2 - W1R2 + R1R2)/R1R2

    xi_pi_sigma = np.zeros((npi,nsi))
    for j in range(nsi):
        for i in range(npi):
            xi_pi_sigma[i,j] = xi2D[i+j*npi]

    wp_marked = 2*np.sum(xi_pi_sigma,axis=0)*dpi

    MCF_JK = (1+wp_marked/rp)/(1+wp_unmarked/rp)
    return MCF_JK
