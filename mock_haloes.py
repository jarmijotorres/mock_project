import numpy as np
from glob import glob
import h5py,sys
from astropy.table import Table

#c200_table = np.loadtxt('/cosma7/data/dp004/dc-armi2/HOD_mocks/concentration_mass/c200c_M200c_L768_z0.3_scatter.dat')
#table of concentration_mass_relation following Klypin+2014 last 2 colums have the 1sigma scatter to recreate concentration of all haloes in the simulation

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

print('Found {} blocks of data in {} directory. Opening data...'.format(len(haloes_name),haloes_dir) )
M200c = []
R200c = []
pos = []
IDs = []
ids = 0
for i in range(len(haloes_name)):#there is normally only 1 block with the useful haloes. Please check before trying anything
    haloes = h5py.File(haloes_name[i],'r')
    M_all = haloes['Group/Group_M_Crit200'][:]*1e10
    R_all = haloes['Group/Group_R_Crit200'][:]
    
    M200c.append(M_all)
    R200c.append(R_all)
    pos.append(haloes['Group/GroupPos'][:])
    ID = np.arange(ids,ids+len(M_all))
    IDs.append(ID)
    ids = ID[-1] + 1

M200c = np.concatenate(M200c)
R200c = np.concatenate(R200c)
pos = np.concatenate(pos)
IDs = np.concatenate(IDs)
print('Data reading done...')
print('Calculating concentration...')

Mhalo_min = 1.6e11

R200c = R200c[M200c>Mhalo_min]
pos = pos[M200c>Mhalo_min]
IDs = IDs[M200c>Mhalo_min]
M200c = M200c[M200c>Mhalo_min]
#c200c = c_of_M_Klypin(M200c,C0,g,M0)#10**(P0[0]*(np.log10(M200c)) + P0[1])
#c200_1s = np.interp(np.log10(M200c ),c200_table[:,3],c200_table[:,9],left=c200_table[0,9],right=c200_table[-1,9])
#c200_scatter = 10**np.random.normal(loc=np.log10(c200c),scale=c200_1s)

print('Done.')
#new  concentration mass relation for catalogue at z=0.3

#write as astropy table:
dir_out = '/cosma7/data/dp004/dc-armi2/HOD_mocks/halo_catalogues/'
cat = Table(data = [IDs,M200c,pos,R200c],names=['ID','M200','pos','R200'])
cat.write(dir_out+'Haloes_MG-Gadget_'+Mod+'_z'+str(z_s)+'_L'+str(Lbox)+'_ID_M200c_pos_R200_logMhalomin_11.2.hdf5', format='hdf5',path='data')
print('saving data in: {}.'.format(dir_out))
print('End of program')
#======= concentration estimation ========#
