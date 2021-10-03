import numpy as np
from glob import glob
import h5py,sys
from astropy.table import Table

def c_of_M_Klypin(M,C0,g,M0):#Klypin et al. (2014) model
    return C0 * (M / 1e12)**(-1*g)*(1+(M/M0)**0.4)
    #return 4.67 * (M / 1e14)**(-0.11)
#C0=7.75; g=0.1; M0=4.5e5#at z=0.0
C0=6.7; g=0.095; M0=2.0e16#at z=0.3

#halo catalogue
Mod = sys.argv[1]
Lbox = int(sys.argv[2])
S = sys.argv[3]#'302'
haloes_dir = sys.argv[4]

redshift_dict = {'302':0.3,'400':0.0,'175':1.0}
z_s = redshift_dict[S]

haloes_name = np.sort(glob(haloes_dir+'fof_subhalo_tab_'+S+'.*.hdf5'))

#print('Found {} blocks of data in {} directory. Opening data...'.format(len(haloes_name),haloes_dir) )
i=0#first block has the most massive objects with M>10^12Msun/h that are more likely to have satellites galaxies
haloes = h5py.File(haloes_name[i],'r')
last_halo_block = haloes['Subhalo/SubhaloGrNr'][-1]
c=0
POS_SH = np.zeros(last_halo_block+1,dtype=object)
M_h = np.zeros(last_halo_block+1,dtype=float)
CM = np.zeros((last_halo_block+1,3),dtype=float)
for j in range(len(POS_SH)):
    M_h[j] = haloes['Group/Group_M_Crit200'][j]
    CM[j] = haloes['Group/GroupCM'][j]
    Nsub = haloes['Group/GroupNsubs'][j]
    pos_subh = haloes['Subhalo/SubhaloPos'][c:c+Nsub]
    c+=Nsub
    POS_SH[j] = pos_subh
IDs = np.arange(last_halo_block+1)
M_h *= 1e10
Mhalo_min = 1e12

CM_large_haloes = CM[M_h>Mhalo_min]
M_large_haloes = M_h[M_h>Mhalo_min]
POS_SH_large_haloes = POS_SH[M_h>Mhalo_min]
IDs_large_haloes = IDs[M_h>Mhalo_min]

#print('Done.')
#new  concentration mass relation for catalogue at z=0.3

#write as astropy table:
dir_out = '/cosma7/data/dp004/dc-armi2/HOD_mocks/halo_catalogues/subhaloes/'
cat = Table(data = [IDs_large_haloes,M_large_haloes,CM_large_haloes,POS_SH_large_haloes],names=['ID','M200c','CM','SubHaloList'])

cat_numpy = np.array(cat)
np.save(dir_out+'Haloes_MG-Gadget_'+Mod+'_z'+str(z_s)+'_L'+str(Lbox)+'_ID_subhalotable_CM_M200c_12.0.npy',cat_numpy,allow_pickle=True)

print('saving data in: {}.'.format(dir_out))
print('End of program')
#======= concentration estimation ========#
