import numpy as np
from scipy.spatial import Voronoi,voronoi_plot_2d

M = 'GR'#sys.argv[1]
Lbox = 768#int(sys.argv[2])

if M == 'GR':
    theta = '13.102_14.077_13.117_0.150_1.011'
elif M == 'F5':
    theta = '13.161_13.982_13.580_0.111_0.991'
l_galaxies = '/cosma7/data/dp004/dc-armi2/HOD_mocks/galaxy_catalogues/Galaxy_'+M+'_z0.3_L'+str(Lbox)+'_HOD_'+theta+'_pos_HaloM200c_weight.dat'

G1 = np.loadtxt(l_galaxies)
id_cen = np.where(G1[:,5]==1)[0]

Haloes_stacked = []
Haloes_stacked.append(np.array([[0.,0.,0.]]))
for id_i in id_cen[:20]:
    gid = id_i + 1
    c=0
    while G1[gid,-1] == -1.:
        c+=1 
        gid += 1
    halo_i = G1[id_i+1:id_i+1+c,(0,1,2)] - G1[id_i,(0,1,2)]
    Haloes_stacked.append(halo_i)    
Haloes_stacked = np.concatenate(Haloes_stacked)

vd_GR = Voronoi(Haloes_stacked_GR[:,(0,1)])
vd_F5 = Voronoi(Haloes_stacked_F5[:,(0,1)])