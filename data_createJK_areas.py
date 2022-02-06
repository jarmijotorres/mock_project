import numpy as np
import h5py
import healpy as hp

NSIDE=1024
LOWZ_mask_IDs = np.loadtxt('/cosma5/data/dp004/dc-armi2/SDSS/BOSS/DR12/mask/randoms_LOWZ_North_NS1024.mask')
ranmask = np.zeros(12*NSIDE**2)
ranmask[LOWZ_mask_IDs.astype(int)] = 1.

pix_angles = hp.pix2ang(nside=1024,ipix=LOWZ_mask_IDs.astype(int),nest=True)
#output format is theta,phi ([0,pi],[0,2pi])

#convert to ra,dec ([0,360],[-90,90])
radec_pixels = np.array([np.rad2deg(pix_angles[1]),np.rad2deg(pix_angles[0]-np.pi/2.)]).T
dec_sorting = np.argsort(radec_pixels[:,1])
dec_divs = np.array_split(radec_pixels[dec_sorting],10)
radec_patches = []
for radec in dec_divs:
    ra_sorting = np.argsort(radec[:,0])
    ra_divs = np.array_split(radec[ra_sorting],10)
    radec_patches.append(ra_divs)
    
limit_areas = np.zeros((100,4))
for i,radec_row in enumerate(radec_patches):
    for j,radec in enumerate(radec_row):
        limit_areas[i+10*j] = np.array([radec[:,0].min(),radec[:,0].max(),radec[:,1].min(),radec[:,1].max()])
        
np.savetxt('/cosma7/data/dp004/dc-armi2/HOD_mocks/survey_galaxies/limits_subvolumes_100.txt',limit_areas)

limit_areas = np.loadtxt('/cosma7/data/dp004/dc-armi2/HOD_mocks/survey_galaxies/limits_subvolumes_100.txt')

hf = h5py.File('/cosma7/data/dp004/dc-armi2/HOD_mocks/galaxy_catalogues/twopcf_catalogues/V2D_8_slices_dz_0.015_galaxy_LOWZ_North_z0.24_0.36.hdf5', 'a')

JK_area = np.zeros(len(hf['Z']))
for rank in range(len(limit_areas)):
    cond1 = (hf['RA']>limit_areas[rank][0])&(hf['RA']<limit_areas[rank][1])&(hf['DEC']>limit_areas[rank][2])&(hf['DEC']<limit_areas[rank][3])
    JK_area[cond1] = rank
    
hf.create_dataset('JK_ID',data=JK_area,dtype=int)
hf.close()