import os

import numpy as np
from numba import jit

import illustris_python as il 
from illustrisfrb import illustrisfrb as ifrb

@jit
def func(xyz, mask, xhalo, rhalo):
    x_p, y_p, z_p = xhalo
    for ii, x in enumerate(xyz):
        for jj, y in enumerate(xyz):
            for kk, z in enumerate(xyz):
                sep = np.sqrt((x - x_p)**2 + (y - y_p)**2 + (z - z_p)**2)
                maskelement = sep < rhalo
                mask[ii,jj,kk] += maskelement
    return mask

def make_halo_mask(nhalo, snapNum):
    outdir = './snap%ddata/' % snapNum

    os.makedirs(outdir, exist_ok=True)

    frb = ifrb.IllustrisFRB("sims.TNG/TNG300-1/output/", snapNum, "basedir")

    halos = frb.read_groups(Mmin=1e12, Mmax=np.inf)
    xyz_halos = halos[0]
    Mhalo = halos[-2].value
    r500 = halos[-1]
    nhalo = min(nhalo, len(r500))

    print("Read in Snap %d" % snapNum)

    xyz = np.linspace(1/3.*width, 2/3*width, 500)

    nbin = len(xyz)
    mask = np.zeros((nbin, nbin, nbin), dtype=bool)

    for nn in range(len(Mhalo))[:nhalo]:        
        xhalo = xyz_halos[nn]
        
        # Assume r500=0.65*r200
        mask += func(xyz, mask, xhalo, (1/0.65) * r500[nn])

        if nn % 100 == 0:
            print(nn)

        if nn % 1000 == 0 and nn > 0:
            print('Writing to disk')
            np.save(outdir + 'mask2r200_%d.npy' % nn, mask)

    return mask