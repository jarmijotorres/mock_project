import numpy as np
import sys
from scipy.spatial import cKDTree
from glob import glob
from astropy.coordinates import SkyCoord
from mpi4py import MPI
comm =MPI.COMM_WORLD

rank = comm.Get_rank()
size = comm.Get_size()

M = sys.argv[1]

nrbins = 10
rbins = np.logspace(np.log10(0.5),np.log10(50.),nrbins+1)
r_b = 10**(np.log10(rbins[:-1]) + np.diff(np.log10(rbins))[0]/2.)#but in real space becasue is fair from the simulation pov
V_mean = 1./(2.8e-4*38.4)
p=-0.5
m_bar = V_mean**p

def dN_dm_per_sep_bin(G0):
    nn_tree = cKDTree(G0[:,(0,1,2)])
    dN_dmark_pairs = np.zeros((nrbins,20))
    m_per_sep_bins = []
    for i in range(nrbins):
        if i == 0:
            pairs_sep_i = nn_tree.query_pairs(r=rbins[i+1])
            pairs_fin = pairs_sep_i 
        else:
            pairs_sep_i = nn_tree.query_pairs(r=rbins[i+1])#
            pairs_fin = pairs_sep_i - pairs_bel

        pairs_bel = pairs_sep_i
        
        mm = np.array(list(pairs_fin))
        
        Npairs_per_sep_bin = len(mm)
        V_i = G0[mm[:,0],3]
        V_unique = np.unique(V_i)
        m_i = (V_unique)**p
    
        Nm,_ = np.histogram(np.log10(m_i/(m_bar)),range=(-1.0,2.0),bins=20)
        dN_dmark_pairs[i] = Nm / Npairs_per_sep_bin
        mb = 0.5*(_[:-1] + _[1:])
        m_per_sep_bins.append(m_i)
    return m_per_sep_bins,np.vstack([mb,dN_dmark_pairs]).T


list_galaxies = glob('/cosma7/data/dp004/dc-armi2/HOD_mocks/galaxy_catalogues/boxes/family/marks/V2D_Galaxy_'+M+'_z0.3_HOD_*_1sigma_0.5Angal_0.5Awp_chi2_20_slices_ssize38.4.vol')

list_all = np.array_split(list_galaxies,28)

list_chunk = list_all[rank]

for l in list_chunk:
    HOD_name = l.split('_')[7:12]
    G0 = np.loadtxt(l)
    S = dN_dmm_per_sep_bin(G0)
    np.savetxt('/cosma7/data/dp004/dc-armi2/HOD_mocks/galaxy_catalogues/boxes/family/marks/histograms/dN_dlogmm_'+M+'_%s_%s_%s_%s_%s_sep_r_0.5_50_10logrbins.dat'%tuple(HOD_name),S)
    
comm.barrier()

if rank == 0:
    print('all files saved by ranks...')