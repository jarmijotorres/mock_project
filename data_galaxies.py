import numpy as np
from astropy.io import fits
import os, sys

### input catalogues for CUTE.
fdir = '/cosma5/data/dp004/dc-armi2/SDSS/BOSS/DR12/data/'
iszcut = True
print("Assuming SDSS data the name of the BOSS subsample (e.g. CMASS_North).\n The corresponding random file will also be opened")
fname = sys.argv[1]
#gor = 'galaxy'
print("opening FITS file"+fdir+'galaxy_DR12v5_'+ fname +'.fits.gz'+". Also opening its random file... \n")
fdata = fits.open(fdir+'galaxy'+'_DR12v5_'+ fname +'.fits.gz')
rdata = fits.open(fdir+'random0'+'_DR12v5_'+ fname +'.fits.gz')

adata = fdata[1].data
bdata = rdata[1].data
Z = adata['Z'] 
rZ = bdata['Z']
if iszcut:
    print("indicate min and max redshift for the limited subsample (e.g. CMASS_North 0.45 0.6)")
    zmin = float(sys.argv[2]); zmax = float(sys.argv[3])
    adata = adata[(Z>zmin)&(Z<zmax)]
    bdata = bdata[(rZ>zmin)&(rZ<zmax)]
    fname += '_z' + str(zmin) + '_' + str(zmax)
RA = adata['RA']; DEC = adata['DEC']; Z = adata['Z']
rRA = bdata['RA']; rDEC = bdata['DEC']; rZ = bdata['Z']; rw0 = bdata['WEIGHT_FKP'];

w0 = adata['WEIGHT_FKP']; w1 = adata['WEIGHT_CP']; w2 = adata['WEIGHT_NOZ']; w3 = adata['WEIGHT_STAR']; w4 = adata['WEIGHT_SEEING']; w5 = adata['WEIGHT_SYSTOT']
wt =(w1+w2-1)* w0*w3*w4
    
S = np.vstack([RA,DEC,Z,wt]).T
print('CUTE format: \ncol1: RA\t col2: DEC\t col3: z\t col4: weight_tot')
fm = '%.8lf %.8lf %.8lf %.8lf'
odir = '/cosma5/data/dp004/dc-armi2/SDSS/BOSS/DR12/subsamples/'
print("file saved in "+odir+'galaxy_'+fname+".dat \n")
np.savetxt(odir+'galaxy'+'_'+fname+'.dat',S,fmt=fm,comments='#')

rS = np.vstack([rRA,rDEC,rZ,rw0]).T
odir = '/cosma5/data/dp004/dc-armi2/SDSS/BOSS/DR12/subsamples/'
print("random saved in "+odir+'random0_'+fname+".dat\n")
np.savetxt(odir+'random0_'+fname+'.dat',rS,fmt=fm,comments='#')
print("... done.\n")


### North + South

#NGC = np.loadtxt('/cosma5/data/dp004/dc-armi2/SDSS/BOSS/DR12/subsamples/galaxy_LOWZ_North_z0.2_0.4.dat')
#SGC = np.loadtxt('/cosma5/data/dp004/dc-armi2/SDSS/BOSS/DR12/subsamples/galaxy_LOWZ_South_z0.2_0.4.dat')

#ngc_sgc = np.vstack([NGC,SGC])
#np.savetxt('/cosma5/data/dp004/dc-armi2/SDSS/BOSS/DR12/subsamples/galaxy_LOWZ_z0.2_0.4.dat',ngc_sgc)