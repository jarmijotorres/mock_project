import numpy as np
from glob import glob
import h5py,sys
from astropy.table import Table

#halo catalogue
Mod = sys.argv[1]
Lbox = int(sys.argv[2])
S = sys.argv[3]#'302'
haloes_dir = sys.argv[4]

redshift_dict = {'302':0.3,'400':0.0,'175':1.0}
z_s = redshift_dict[S]

haloes_name = np.sort(glob(haloes_dir+'fof_subhalo_tab_'+S+'.*.hdf5'))
print('Found {} blocks of data in {} directory. Opening data...'.format(len(haloes_name),haloes_dir) )

M_all = []
CM_all = []
R_all = []
pos_all = []
IDs_all = []
Nsub_all = []
sh_GrNr_all = []
sh_pos = []
sh_mass = []
FirstSub_all = []
#we set a min mass to select haloes. Normally the limit where the low-res run stops producing haloes Mmin = 1.6e11
id_i = 0
for hi in haloes_name:
    #main halo information
    haloes = h5py.File(hi,'r')
    M = haloes['Group/Group_M_Crit200'][:]
    ID = np.arange(id_i,id_i+len(M))
    id_i += len(M)
    M_nonzero = M[M>16][:]*1e10
    ID_nonzero = ID[M>16]
    R = haloes['Group/Group_R_Crit200'][:][M>16]
    pos = haloes['Group/GroupPos'][:][M>16]
    Nsub = haloes['Group/GroupNsubs'][:][M>16]
    CM = haloes['Group/GroupCM'][:][M>16]
    group_firstsub = haloes['Group/GroupFirstSub'][:][M>16]
    M_all.append(M_nonzero)
    R_all.append(R)
    pos_all.append(pos)
    IDs_all.append(ID_nonzero)
    Nsub_all.append(Nsub)
    CM_all.append(CM)
    FirstSub_all.append(group_firstsub)
    #subhalo information
    sh_GrNr_all.append(haloes['Subhalo/SubhaloGrNr'])
    sh_pos.append(haloes['Subhalo/SubhaloPos'])
    sh_mass.append(haloes['Subhalo/SubhaloMass'][:]*1e10)
    
M_all = np.concatenate(M_all)
R_all = np.concatenate(R_all)
CM_all = np.concatenate(CM_all)
pos_all = np.concatenate(pos_all)
IDs_all = np.concatenate(IDs_all)
Nsub_all = np.concatenate(Nsub_all)
FirstSub_all = np.concatenate(FirstSub_all)
#
sh_GrNr_all = np.concatenate(sh_GrNr_all)
sh_pos_all = np.concatenate(sh_pos)
sh_mass_all = np.concatenate(sh_mass)
#ignore subhalos without the required haloes

#sorting subhaloes
print('sorting subhaloes in haloes...')

full_sh_id_mass = []
for i in range(len(M_all)):
    ID_sh = sh_GrNr_all[FirstSub_all[i]:FirstSub_all[i]+Nsub_all[i]]
    pos_sh = sh_pos_all[FirstSub_all[i]:FirstSub_all[i]+Nsub_all[i]]
    mass_sh = sh_mass_all[FirstSub_all[i]:FirstSub_all[i]+Nsub_all[i]]
    full_sh_id_mass.append(np.vstack([ID_sh,pos_sh.T,mass_sh]).T)
full_sh_id_mass = np.concatenate(full_sh_id_mass) 

first_sh = np.unique(full_sh_id_mass[:,0],return_index=True)[1]
    #[FirstSub_all[i]]
    
#sh_table = np.zeros(len(IDs_all),dtype=object)
#sh_table_mass = np.zeros(len(IDs_all),dtype=object)
inter_sh_IDs = np.intersect1d(sh_GrNr_all,IDs_all,return_indices=True)

#for i,id_i in enumerate(inter_sh_IDs[2]):
#    hi_grid = np.zeros((Nsub_all[id_i]-1,3))
#    hi_mass_grid = np.zeros((Nsub_all[id_i]-1,1))
#    sh_in_grid = sh_pos_all[inter_sh_IDs[1][i]:inter_sh_IDs[1][i]+Nsub_all[id_i]]
#    sh_mass_in_grid = sh_mass_all[inter_sh_IDs[1][i]:inter_sh_IDs[1][i]+Nsub_all[id_i]]
#    hi_grid = sh_in_grid[1:]
#    hi_mass_grid = sh_mass_in_grid[1:]
#    sh_table[id_i] = hi_grid
#    sh_table_mass[id_i] = hi_mass_grid
#print('Done.')

#full_sh_id_mass = []
#for i,id_i in enumerate(inter_sh_IDs[2]):
#    hi_grid = np.zeros((Nsub_all[id_i]-1,3))
#    hi_mass_grid = np.zeros((Nsub_all[id_i]-1,1))
#    sh_in_grid = sh_pos_all[inter_sh_IDs[1][i]:inter_sh_IDs[1][i]+Nsub_all[id_i]]
#    sh_mass_in_grid = sh_mass_all[inter_sh_IDs[1][i]:inter_sh_IDs[1][i]+Nsub_all[id_i]]
#    hi_grid = sh_in_grid[1:]
#    hi_mass_grid = sh_mass_in_grid[1:]
#    sh_group = np.vstack([np.full(len(hi_grid),inter_sh_IDs[0][i]),hi_grid.T,hi_mass_grid]).T
#    full_sh_id_mass.append(sh_group)
#full_sh_id_mass = np.concatenate(full_sh_id_mass)
 
dir_out = '/cosma7/data/dp004/dc-armi2/HOD_mocks/halo_catalogues/'
cat_mainhaloes = Table(data = [IDs_all,M_all,R_all,CM_all,pos_all,Nsub_all,first_sh],names=['ID','M200c','R200c','CM','pos','Nsh','FirstSh'])
cat_subhaloes = Table(data = [full_sh_id_mass[:,0],full_sh_id_mass[:,(1,2,3)],full_sh_id_mass[:,4]],names=['ID_MainHalo','pos','M200c'])

f = h5py.File(dir_out+'Haloes_MG-Gadget_'+Mod+'_z'+str(z_s)+'_L'+str(Lbox)+'_ID_M200c_R200c_pos_Nsh_FirstSh_SubHaloList_SubHaloMass_logMmin_11.2.0.hdf5', 'w')
f.create_dataset('MainHaloes', data=cat_mainhaloes)
f.create_dataset('SubHaloes', data=cat_subhaloes)

f.close()

#p.save(dir_out+'Haloes_MG-Gadget_'+Mod+'_z'+str(z_s)+'_L'+str(Lbox)+'_ID_M200c_R200c_pos_Nsh_SubHaloList_SubHaloMass_logMmin_11.2.0.npy',cat_numpy,allow_pickle=True)
print('saving data in: {}.'.format(dir_out))
print('End of program')

