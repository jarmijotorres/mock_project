import numpy as np
import pandas as pd
import pyvoro,sys
from mpi4py import MPI
comm =MPI.COMM_WORLD

rank = comm.Get_rank()
size = comm.Get_size()

voro_dir = '/cosma7/data/dp004/dc-armi2/HOD_mocks/marks/'
l = sys.argv[1]
cat_name = l.split('/')[-1]
M = 'GR'#sys.argv[2]
Lbox = 768#int(sys.argv[3])
#int(sys.argv[4])

ls = Lbox/float(size) #length of each slice

G1 = np.loadtxt(l,usecols=(0,1,2))
zi = rank*ls
zf = (rank+1)*ls
G1_chunk = G1[:,(0,1,2)][(G1[:,2]>zi)&(G1[:,2]<=zf)]

G1_unique, unique_index, unique_inverse = np.unique(G1_chunk,axis=0,return_index=True,return_inverse=True)
#is_unique = np.zeros(len(G1_chunk),dtype=int)
#is_unique[unique_index] = 1  

V2D = pyvoro.compute_2d_voronoi(points=G1_unique[:,(0,1)],limits=[[0.,Lbox],[0.,Lbox]],dispersion=50.0,periodic=[True,True],)
    #
voronoi_df = pd.DataFrame(data=V2D)
#voronoi_df.to_csv(voro_dir+l+'.df.csv')
vol2D = voronoi_df['volume'].values
del V2D

vol2D_chunk = vol2D[unique_inverse]

comm.barrier()
vols_all = comm.gather(vol2D_chunk,root=0)
G1_all = comm.gather(G1_chunk,root=0)
if rank ==0:
    vols_all = np.concatenate(vols_all)
    G1_all = np.concatenate(G1_all,axis=0)
    G1_vol = np.vstack([G1_all.T,vols_all]).T
    np.savetxt(voro_dir+'V2D_'+cat_name+'.%d_slices_ssize%.1lf.vol'%(size,ls),G1_vol)

#S = np.vstack([G1.T,vols_all.T]).T
#np.savetxt(voro_dir+l+'_slices.txt.vol',S)
#open volumes and save galaxy + mark
