# Tools for FRB DMs in IllustrisTNG
# Date : November 2022
# Author : Liam Connor, liam.dean.connor@gmail.com
import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

from astropy import units as u
from astropy import constants as const
import pandas as pd 
import h5py
import glob
from scipy import interpolate

import illustris_python as il 

plt.rcParams.update({
                    'font.size': 12,
                    'font.family': 'serif',
                    'axes.labelsize': 14,
                    'axes.titlesize': 15,
                    'xtick.labelsize': 12,
                    'ytick.labelsize': 12,
                    'xtick.direction': 'in',
                    'ytick.direction': 'in',
                    'xtick.top': True,
                    'ytick.right': True,
                    'lines.linewidth': 0.5,
                    'lines.markersize': 5,
                    'legend.fontsize': 14,
                    'legend.borderaxespad': 0,
                    'legend.frameon': True,
                    'legend.loc': 'lower right'})

colors1 = ['k', '#482677FF', '#238A8DDF', '#95D840FF']

class IllustrisFRB:
    def __init__(self, basepath, snapNum, basedir):
        self.basePath = basepath
        self.snapNum = snapNum
        self.basedir = basedir

        self.snapfields = ['PartType0/Masses',
                           'PartType0/Coordinates',
                           'PartType0/ElectronAbundance',
                           'PartType0/Density',
                           'PartType0/StarFormationRate']

        self.subhalo_fields = ['SubhaloMass',
                             'SubhaloSFRinRad',
                             'SubhaloPos',
                             'SubhaloHalfmassRad',
                             'SubhaloMassInHalfRad',
                             'SubhaloStellarPhotometrics']

        self.xyz_origin = np.array([0,0,0])

    def get_header(self):
        header = il.groupcat.loadHeader(self.basePath, self.snapNum)
        redshift = header['Redshift']
        boxsize_kpc = header['BoxSize']

        return header

    def cart2spher(self,x,y,z):
        xy = x**2 + y**2
        r = np.sqrt(xy + z**2)
        theta = np.arctan2(np.sqrt(xy), z) # for elevation angle defined from Z-axis down
        #ptsnew[:,4] = np.arctan2(xyz[:,2], np.sqrt(xy)) # for elevation angle defined from XY-plane up
        phi = np.arctan2(y, x)
        return r,theta,phi


    def spher2cart(self,r,theta,phi):
        x = r*np.cos(phi)*np.sin(theta)
        y = r*np.sin(phi)*np.sin(theta)
        z = r*np.cos(theta)

        return x,y,z

    def spher2cart_(self,r,theta,phi):
        x = r*np.sin(phi)*np.cos(theta)
        y = r*np.sin(phi)*np.sin(theta)
        z = r*np.cos(theta)

        return x,y,z

    def dm_cell(self, frac_e, density, 
                cellsize, bperp, X_H=0.75):
        """ Get DM for each cell
        """
        mH = 1.67262192e-24 * u.g # grams
        Ne = X_H * frac_e * density * 1e10 * u.M_sun / u.kpc**3 / mH 
        Ne = Ne.to(u.cm**-3)
        dr = 2 * np.sqrt(cellsize**2 - bperp**2)
        dmarr = Ne * dr
        dm = Ne * dr
        dm = dm.to(u.pc*u.cm**-3)
        return dm, 1e3*dmarr, dr 

    def select_halos(self, subhalos, Mmin=0, 
                           Mmax=np.inf, field_gal=False,
                           xyz_halos=None):
        Mhalo = subhalos['SubhaloMass'] * 1e10
        ind = np.where((Mhalo > Mmin) & (Mhalo < Mmax))[0]

        if field_gal is True:
            assert xyz_halos is not None, "Need xyz_halos if field_gal=True"
            xyz_halos = subhalos['SubhaloPos']
            xyz_massive = xyz_halos[Mhalo>5e13]
            ind_field = []
            for ii in ind:
                sep_kpc = np.sqrt(np.sum(np.abs(xyz_halos[ii]-xyz_massive)**2, axis=1))
                if sep_kpc.min() > 2000.0:
                    ind_field.append(ii)
            ind = ind_field            

        return ind

    def read_snapchunk(self, file=None, fn='simulation.hdf5', 
                       snapfields=None,
                       start=0, snapNum=98,
                       stop=int(1e8), calc_volume=True):
        """ Read in a chunk of snapshot data between cell 
        index start and stop. 
        """
        if file is None:
            f = h5py.File(fn,'r')
        else:
            f = file 

        if snapfields is None:
            snapfields = self.snapfields

        data_dict = {}

        for field in snapfields:
            data = f['/Snapshots/%d/%s'%(snapNum,field)][start:stop]
            data_dict.update({field:data})

        if calc_volume:
            # Convert to solar masses 
            masses = data_dict['PartType0/Masses'] * u.M_sun * 1e10 
            # Convert to solar masses per kpc**3
            density = data_dict['PartType0/Density'] * u.M_sun*1e10 / u.kpc**3
            volume = masses / density
            volume[np.isnan(volume)] = 0.0
            volume = volume.value
            data_dict.update({'PartType0/Volume':volume})
            cellsize = (volume * 3.0 / (4*np.pi))**(1.0/3.0)
            data_dict.update({'PartType0/Cellsize':cellsize})

        return data_dict

    def read_halos(self, Mmin=0, Mmax=np.inf, field_gal=False):
        """ Read in halos from subhalo catalog 
        """
        subhalos = il.groupcat.loadSubhalos(self.basePath,
                                            self.snapNum,
                                            fields=self.subhalo_fields)
        ind_halos = self.select_halos(subhalos, Mmin=Mmin, 
                                       Mmax=Mmax, field_gal=field_gal)
        
        Mhalo = subhalos['SubhaloMass']*1e10*u.M_sun
        Mhalo = Mhalo[ind_halos]
        subhalos_r200 = subhalos['SubhaloHalfmassRad'][ind_halos]#[ind_large]
        xyz_halos = subhalos['SubhaloPos'][ind_halos]

        r_halo, theta_halo, phi_halo = self.cart2spher(xyz_halos[:,0],
                                                  xyz_halos[:,1],
                                                  xyz_halos[:,2])

        return xyz_halos, r_halo, theta_halo, phi_halo, Mhalo, subhalos_r200

    def compute_dm_los(self,xyz,xyz_frb,dm_cell,cellsize):

        print("Assuming cylinder is along z-axis")
        sep_kpc = np.sqrt((xyz_frb[0]-xyz[:,0])**2 + (xyz_frb[1]-xyz[:,1])**2)
        ind = np.where(sep_kpc<cellsize)[0]

        adjust_los = np.sqrt(cellsize[ind]**2-sep_kpc[ind]**2) / sep_kpc[ind]
        dm_cell_tangent = dm_cell[ind] * adjust_los
        dr_los = cellsize[ind]

        return dr_los, dm_cell_tangent, xyz[ind, 2]

    def compute_frb_los_halo(self, xyz, cellsize, frac_e, density, 
                            thetas_frb, phis_frb, r_frb,
                            nlos=1, save_csv=False, fnout=''):
        """ Find the cells that are intersected by 
        FRB lines of sight. Calculate the DM along 
        those LoS.
        """

        x,y,z = xyz[:,0],xyz[:,1],xyz[:,2]
        x = x# - boxsize/2.
        y = y# - boxsize/2.
        z = z# - boxsize/2.
        r_cell, theta_cell, phi_cell = self.cart2spher(x,y,z)

        dm_los = 0
        DM = np.zeros([nlos]) 
        Dist_cell = np.zeros([nlos])
        Dist_tot = np.zeros([nlos])
        Ne_arr = np.zeros([nlos])
        n_halo_intersect = np.zeros([nlos])

        mH = 1.67262192e-24 * u.g # grams

        for ii in range(nlos):
            print("        LoS %d/%d" % ((1+ii),nlos))
            theta, phi = thetas_frb[ii], phis_frb[ii]
            sep_cell = np.sqrt(((theta_cell-theta)*np.cos(phi))**2 + (phi_cell-phi)**2.)
            bperp = r_cell * sep_cell
            # Check where cell is close to LoS but not in the galaxy disk
            ind = (bperp < cellsize.value) & (r_cell-r_frb[ii]>10.0)
            if np.sum(ind)==0:
                print("Skipping because there are no nearby cells")
                continue

            dm_los, dr = self.dm_cell(frac_e[ind], density[ind], cellsize[ind], bperp[ind])
            dm = np.sum(dm_los).to(u.pc*u.cm**-3)
            rtot = r_cell[ind].max()-r_cell[ind].min()
            rlos = r_cell[ind]-r_cell[ind].min()

    #        print(r_cell[ind]-r_frb[ii])
    #        print(Ne*dr)
            rs = np.linspace(r_cell[ind].min(), r_cell[ind].max(), 1000)
            DM[ii] = dm.value
            Dist_cell[ii] = np.sum(dr.value)
            Dist_tot[ii] = rtot
            x_frb,y_frb,z_frb = spher2cart(rs, theta, phi)

        FRBarr = np.empty([nlos, 5])
        FRBarr[:, 0] = DM
        FRBarr[:, 1] = thetas_frb
        FRBarr[:, 2] = phis_frb
        FRBarr[:, 3] = Dist_cell
        FRBarr[:, 4] = Dist_tot

        FRBarr_pd = pd.DataFrame(FRBarr, columns=['dm',
                                                  'theta_rad',
                                                  'phi_rad', 
                                                  'dist_cell', 
                                                  'dist_tot']
                                                   )
        if save_csv:
            FRBarr_pd.to_csv('FRBsightlines_%s.csv' % fnout)

        return FRBarr_pd, dm_los, rlos

    def get_gas_properties(self, data, xyz, ind):
        sfr = data['PartType0/StarFormationRate'][ind]
        nosfr = sfr == 0 

        xyz_ind = xyz[ind][nosfr]
        # Size of cell in kpc 
        cellsize = data['PartType0/Cellsize'][ind][nosfr]*u.kpc
        # Electron number density 
        ne = data['PartType0/ElectronAbundance'][ind][nosfr]
        # Total gas density of cell
        density = data['PartType0/Density'][ind][nosfr]

        return xyz_ind, cellsize, ne, density

    def cluster_dm_profile(self,nhalo=10, nchunkuse=1, plot_halo=False):
        _ = self.read_halos(Mmin=1e14,Mmax=np.inf,field_gal=False)
        xyz_halos, r_halo, theta_halo, phi_halo, Mhalo, subhalos_r200 = _

        f = h5py.File('simulation.hdf5','r')
        dfull = 0
        
        dphis = np.deg2rad(np.linspace(-0.5, 0.5, 50))
        dm = np.zeros([nhalo, len(dphis)])
        bperp_arr = np.zeros([nhalo, len(dphis)])

        # Cannot read in full dataset, need to read it in chunks
        for chunk in range(nchunkuse):
            print("Processing chunk: %d/%d" % (chunk,nchunkuse))
            
            data=self.read_snapchunk(snapfields=self.snapfields,
                                    start=int(chunk*1e8),
                                    stop=int((chunk+1)*1e8),
                                    calc_volume=True, file=f, 
                                    snapNum=self.snapNum,
                                    )
            return data
            # Coordinates of cells in Cartesian
            xyz = data['PartType0/Coordinates']

            for nn in range(nhalo):
                sep_kpc = np.sqrt(np.sum(np.abs(xyz_halos[nn] - xyz)**2, axis=1))
                ind_clust = np.where(sep_kpc < 3000.0)[0]

                if len(ind_clust)==0:
                    print("Skipping this cluster.")
                    continue
                else:
                    print(len(ind_clust))

                xyz_cluster, cellsize, ne, density = self.get_gas_properties(data, xyz, 
                                                                            ind_clust)
    def read_cylinder(self, xyz_halo, nchunk=1):
        # Cannot read in full dataset, need to read it in chunks

        assert len(xyz_halo)==3, "Expecting just one 3D coordinate"

        f = h5py.File('simulation.hdf5','r')

        outdir = './data_%d-%d'%(xyz_halo[0], xyz_halo[1])

        # Check if the directory exists
        if not os.path.exists(outdir):
            # Create the directory
            os.makedirs(outdir)

        for chunk in range(nchunk)[:]:
            print("Processing chunk: %d/%d" % (chunk,nchunk))
            
            data=self.read_snapchunk(snapfields=self.snapfields,
                                start=int(chunk*3e8),stop=int((chunk+1)*3e8),
                                calc_volume=True, file=f, 
                                snapNum=self.snapNum,
                                )
            xyz = data['PartType0/Coordinates']

            if len(xyz)==0:
                print("    Empty chunk")
                continue

            sep_kpc = np.sqrt(np.sum(np.abs(xyz_halo[:2] - xyz[:, :2])**2, axis=1))
            ind_cyl = np.where(sep_kpc < 5000.0)[0]

            if len(ind_cyl)==0:
                print("    Nothing in cylinder")
                continue

            xyz_cyl, cellsize, ne, density = self.get_gas_properties(data, 
                                                                     xyz, 
                                                                     ind_cyl)
            print(cellsize.max())

            dmtot, dm_cyl, dr  = self.dm_cell(ne, density, 
                                              cellsize, 
                                              0*cellsize, X_H=0.75)

            np.save(outdir+'/xyz_cell_chunk%d'%chunk, xyz_cyl)
            np.save(outdir+'/dm_cell_chunk%d'%chunk, dm_cyl.value)
            np.save(outdir+'/cellsize_chunk%d'%chunk, cellsize.value)

            del dm_cyl, xyz_cyl, ind_cyl, data, xyz

    def get_halo_cells(self, xyz, halo, rthresh):
        xyz_halo = halo[0]
        sep_kpc = np.sqrt(np.sum(np.abs(xyz_halo - xyz)**2, axis=1))
        ind_cyl = np.where(sep_kpc < rthresh)[0]
        return xyz[ind_cyl], ind_cyl

def get_dm_profile(xyz, xyz_halo, DM=None, 
                   cellsize=None, nr=25, ntheta=10):
    frb = IllustrisFRB("output/", 98, "basedir")

    rs = np.linspace(-2000, 2000, nr)
    thetas = np.linspace(0, 2*np.pi, ntheta)

    if DM is None or cellsize is None:
        xyz_cyl, zlos, dm_los, DM, cellsize = dm_from_cyl(xyz_halo, 
                                                      outdir=None, 
                                                      sav=False, 
                                                      xlos=None)

    dm_of_b = np.zeros([ntheta, nr])

    for ii, rr in enumerate(rs):
        for jj, theta in enumerate(thetas):
            x0, y0 = 0, rr
            x_new = x0 * np.cos(theta) - y0 * np.sin(theta)
            y_new = x0 * np.sin(theta) + y0 * np.cos(theta)
            eps = np.array([x_new, y_new, 0])
            print(x_new, y_new, np.sqrt(np.sum(eps**2)))
            dr_los, dm_los, zlos = frb.compute_dm_los(xyz,
                                                      xyz_halo+eps, 
                                                      DM, cellsize)
            dm_of_b[jj,ii] = np.sum(dm_los)

    return rs, dm_of_b



    
def dm_from_cyl(xfrb, outdir=None, sav=True, xlos=None):
    FRBIl = IllustrisFRB("output/", 98, "basedir")

    if xlos is None:
        xlos = xfrb

    if outdir is None:
        outdir = './data_%d-%d'%(xfrb[0], xfrb[1])

    fl = glob.glob(outdir+'/xyz_cell_chunk*')
    fl.sort()
    fldm = glob.glob(outdir+'/dm_cell_chunk*')
    fldm.sort()
    flr = glob.glob(outdir+'/cellsize_chunk*')
    flr.sort()

    xyz_cyl, DM, cellsize = [], [], [] 

    for ii, ff in enumerate(fl):
        print(ii, ff)
        d = np.load(ff)
        dm = np.load(fldm[ii])
        r = np.load(flr[ii])
        xyz_cyl.append(d)
        DM.append(dm)
        cellsize.append(r)

    xyz_cyl = np.concatenate(xyz_cyl)
    cellsize = np.concatenate(cellsize)
    DM = np.concatenate(DM)

    dr_los, dm_los, zlos = FRBIl.compute_dm_los(xyz_cyl,xlos,DM,cellsize)
    sort_index = np.argsort(zlos)

    zlos = zlos[sort_index] 
    dm_los = dm_los[sort_index]

    if sav:
        np.save(outdir+'/DM_los.npy', np.concatenate([zlos, 
                                               dm_los]))
        np.save(outdir+'/xyz_cyl.npy', xyz_cyl)

    return xyz_cyl, zlos, dm_los, DM, cellsize

if __name__=='__main__':
    frb = IllustrisFRB("output/", 98, "basedir")
    plot_halos_paper(cmap='plasma',sep_thresh=2500)
    #halos = frb.read_halos(Mmin=4.8e14, Mmax=4.85e14)
    #xyz_cyl, zlos, dm_los = dm_from_cyl(halos[0][0]+np.array([300,300,0]))
    
#    halos = frb.read_halos(Mmin=1e14, Mmax=np.inf)
    # for ii in range(len(halos[0])):
    #     frb.read_cylinder(halos[0][ii], 50)
    #     xyz_cyl, zlos, dm_los = dm_from_cyl(halos[0][ii])
#    dm_animation(xyz_cyl, zlos, dm_los)
#    a.read_cylinder([112000, 110500, 150000], 100)
#    d = my_frb.cluster_dm_profile(1,1)





