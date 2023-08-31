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


class IllustrisFRB:
    def __init__(self, basepath, snapNum, basedir, fnsim='simulation.hdf5'):
        self.basePath = basepath
        self.snapNum = snapNum
        self.basedir = basedir
        self.fnsim = fnsim

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

        self.group_fields = ['GroupPos',
                             'Group_M_Crit200',
                             'Group_M_Crit500',
                             'Group_R_Crit200',
                             'Group_R_Crit500']

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

    def select_groups(self, groups, Mmin=0, 
                           Mmax=np.inf, field_gal=False,
                           xyz_halos=None):
        Mhalo = groups['Group_M_Crit500'] * 1e10
        ind = np.where((Mhalo > Mmin) & (Mhalo < Mmax))[0]

        if field_gal is True:
            assert xyz_halos is not None, "Need xyz_halos if field_gal=True"
            xyz_halos = groups['GroupPos']
            xyz_massive = xyz_halos[Mhalo>5e13]
            ind_field = []
            for ii in ind:
                sep_kpc = np.sqrt(np.sum(np.abs(xyz_halos[ii]-xyz_massive)**2, axis=1))
                if sep_kpc.min() > 2000.0:
                    ind_field.append(ii)
            ind = ind_field            

        return ind

    def read_snapchunk(self, file=None,  
                       snapfields=None,
                       start=0, snapNum=98,
                       stop=int(1e8), calc_volume=True):
        """ Read in a chunk of snapshot data between cell 
        index start and stop. 
        """
        fn = self.fnsim

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

    def read_subhalos(self, Mmin=0, Mmax=np.inf, field_gal=False):
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

    def read_groups(self, Mmin=0, Mmax=np.inf, field_gal=False):
        """ Read in halos from subhalo catalog 
        """
        groups = il.groupcat.loadHalos(self.basePath,
                                            self.snapNum,
                                            fields=self.group_fields)

        ind_halos = self.select_groups(groups, Mmin=Mmin, 
                                       Mmax=Mmax, field_gal=field_gal)
        
        Mhalo = groups['Group_M_Crit500']*1e10*u.M_sun
        Mhalo = Mhalo[ind_halos]
        subhalos_r200 = groups['Group_R_Crit500'][ind_halos]#[ind_large]
        xyz_halos = groups['GroupPos'][ind_halos]

        r_halo, theta_halo, phi_halo = self.cart2spher(xyz_halos[:,0],
                                                     xyz_halos[:,1],
                                                  xyz_halos[:,2])

        return xyz_halos, r_halo, theta_halo, phi_halo, Mhalo, subhalos_r200

    def compute_dm_los(self,xyz,xyz_frb,dm_cell,cellsize):
        sep_kpc = np.sqrt((xyz_frb[0]-xyz[:,0])**2 + (xyz_frb[1]-xyz[:,1])**2)
        ind = np.where(sep_kpc<cellsize)[0]

        adjust_los = np.sqrt(cellsize[ind]**2-sep_kpc[ind]**2) / sep_kpc[ind]
        dm_cell_tangent = dm_cell[ind] * adjust_los
        dr_los = cellsize[ind]

#        print(np.sum(dr_los * adjust_los))

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
        _ = self.read_subhalos(Mmin=1e14,Mmax=np.inf,field_gal=False)
        xyz_halos, r_halo, theta_halo, phi_halo, Mhalo, subhalos_r200 = _

        f = h5py.File(self.fnsim,'r')
        
        dphis = np.deg2rad(np.linspace(-0.5, 0.5, 50))
        dm = np.zeros([nhalo, len(dphis)])
        bperp_arr = np.zeros([nhalo, len(dphis)])

        # Cannot read in full dataset, need to read it in chunks
        for chunk in range(nchunkuse):
            print("Processing chunk: %d/%d" % (chunk,nchunkuse))
            
            data=self.read_snapchunk(snapfields=self.snapfields,
                                    start=int(chunk*1e8),
                                    stop=int((chunk+1)*1e8),
                                    calc_volume=calc_volume, file=f, 
                                    snapNum=self.snapNum,
                                    )
            return data

    def read_cylinder(self, xyz_halo, nchunk=1, cyl_radius=5000, calc_volume=True):
        # Cannot read in full dataset, need to read it in chunks

        assert len(xyz_halo)==3, "Expecting just one 3D coordinate"

        f = h5py.File(self.fnsim,'r')

        outdir = './data_50e3kpc_%d-%d'%(xyz_halo[0], xyz_halo[1])

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
            ind_cyl = np.where(sep_kpc < cyl_radius)[0]

            if len(ind_cyl)==0:
                print("    Nothing in cylinder")
                continue

            xyz_cyl, cellsize, ne, density = self.get_gas_properties(data, 
                                                                     xyz, 
                                                                     ind_cyl)

            dmtot, dm_cyl, dr  = self.dm_cell(ne, density, 
                                              cellsize, 
                                              0*cellsize, X_H=0.75)

            np.save(outdir+'/xyz_cell_chunk%d'%chunk, xyz_cyl)
            np.save(outdir+'/dm_cell_chunk%d'%chunk, dm_cyl.value)
            np.save(outdir+'/cellsize_chunk%d'%chunk, cellsize.value)

            del dm_cyl, xyz_cyl, ind_cyl, data, xyz

    def read_data_downsample(self, nchunk=1,
                            calc_volume=True, 
                            particletype=0, nbin=1024):
        # Cannot read in full dataset, need to read it in chunks

        f = h5py.File(self.fnsim,'r')

        outdir = './'

        # Check if the directory exists
        if not os.path.exists(outdir):
            # Create the directory
            os.makedirs(outdir)

        density_field = 0 

        for chunk in range(nchunk)[::-1][:]:
            print("Processing chunk: %d/%d" % (chunk,nchunk))
            
            data=self.read_snapchunk(snapfields=self.snapfields,
                                start=int(chunk*5e7),stop=int((chunk+1)*5e7),
                                calc_volume=calc_volume, file=f, 
                                snapNum=self.snapNum,
                                )

            xyz = data['PartType%s/Coordinates' % particletype]
            mass = data['PartType%s/Masses' % particletype]

            if len(xyz)==0:
                print("    Empty chunk")
                continue

            X = np.histogramdd(xyz, bins=(nbin, nbin, nbin), 
                               range=((0, 205e3),(0, 205e3),(0, 205e3)),
                               weights=mass)

            density_field += X[0]

            field_coords = X[1]

        return density_field, field_coords

    def get_halo_cells(self, xyz, halo, rthresh):
        xyz_halo = halo[0]
        sep_kpc = np.sqrt(np.sum(np.abs(xyz_halo - xyz)**2, axis=1))
        ind_cyl = np.where(sep_kpc < rthresh)[0]
        return xyz[ind_cyl], ind_cyl

def get_dm_profile(xyz, xyz_halo, DM=None, 
                   cellsize=None, nr=25, ntheta=10, 
                   sep_thresh=2500.):
    frb = IllustrisFRB("output/", 98, "basedir")

    rs = np.linspace(-sep_thresh, sep_thresh, nr)
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

def estimate_figm(fnbaryon, fnmask):
    mask = np.load(fnmask)
    mass = np.load(fnbaryon)

    mass_in_halos = np.sum(mass[mask])
    mass_in_IGM = np.sum(mass[~mask])

    return mass_in_halos, mass_in_IGM

def dm_from_cyl(outdir=None, sav=True, xlos=None, 
                dochunks=True, nlosx=25, nlosy=25):
    FRBIl = IllustrisFRB("output/", 98, "basedir")

    if outdir is None:
        outdir = './data_full%d-%d'%(xlos[0], xlos[1])

    print("Reading from %s" % outdir)

    fl = glob.glob(outdir+'/xyz_cell_chunk*')
    fl.sort()
    fldm = glob.glob(outdir+'/dm_cell_chunk*')
    fldm.sort()
    flr = glob.glob(outdir+'/cellsize_chunk*')
    flr.sort()

    xyz_cyl, DM, cellsize = [], [], [] 

    if dochunks: 
        zlosarr=[]
        dmlosarr=[]

    for ii in range(len(fl))[:]:
        ff = fl[ii]
        print(ii, ff)
        xyz_cyl_ii = np.load(ff)
        dm = np.load(fldm[ii])
        cellsizeii = np.load(flr[ii])

        if dochunks:
            for xx in np.linspace(51000, 149000, nlosx):
                for yy in np.linspace(51000, 149000, nlosy):
                    dr_los, dm_los, zlos = FRBIl.compute_dm_los(xyz_cyl_ii,[xx,yy,0],dm,cellsizeii)

                    if len(dm_los):
                        np.save("losdata2/z_chunk%d_%d-%d-all.npy"%(ii, xx, yy), zlos)
                        np.save("losdata2/dm_chunk%d_%d-%d-all.npy"%(ii, xx, yy), dm_los)

                    # dr_los, dm_los, zlos = FRBIl.compute_dm_los(xyz_cyl_ii,[0,xx,yy],dm,cellsizeii)

                    # if len(dm_los):
                    #     np.save("losdatax/z_chunk%d_all-%d-%d.npy"%(ii, xx, yy), zlos)
                    #     np.save("losdatax/dm_chunk%d_all-%d-%d.npy"%(ii, xx, yy), dm_los)

                    # dr_los, dm_los, zlos = FRBIl.compute_dm_los(xyz_cyl_ii,[xx,0,yy],dm,cellsizeii)

                    # if len(dm_los):
                    #     np.save("losdatax/z_chunk%d_%d-all-%d.npy"%(ii, xx, yy), zlos)
                    #     np.save("losdatax/dm_chunk%d_%d-all-%d.npy"%(ii, xx, yy), dm_los)                        
        else:
            xyz_cyl.append(xyz_cyl_ii)
            cellsize.append(cellsizeii)
            DM.append(dm)
    
    if dochunks:
        zz = np.concatenate(zlosarr)
        dm = np.concatenate(dmlosarr)
        if sav:
            np.save(outdir+'/DM_los.npy', np.concatenate([zz, 
                                                          dm]).reshape(2,-1))

        return zlosarr, dmlosarr

    xyz_cyl = np.concatenate(xyz_cyl)
    cellsize = np.concatenate(cellsize)
    DM = np.concatenate(DM)

    if xlos is None:
        xlos = np.mean(xyz_cyl,0) + np.array([500,0,0])

    dr_los, dm_los, zlos = FRBIl.compute_dm_los(xyz_cyl,xlos,DM,cellsize)
    sort_index = np.argsort(zlos)

    zlos = zlos[sort_index] 
    dm_los = dm_los[sort_index]

    if sav:
        np.save(outdir+'/DM_los.npy', np.concatenate([zlos, 
                                               dm_los]))
        np.save(outdir+'/xyz_cyl.npy', xyz_cyl)

    return xyz_cyl, zlos, dm_los, DM, cellsize

def get_lots_of_sightlines(outdir,n=50):
    fdirs = glob.glob(outdir)
    DMarr = np.zeros([n,n,len(fdirs)])
    h = 100/69.
    for kk,fd in enumerate(fdirs):
        xyz_cyl, zlos, dm_los, DM, cellsize = dm_from_cyl(outdir=fd, 
                                                          sav=False, 
                                                          xlos=None)
        xs = np.linspace(xyz_cyl[:,0].min()+250, xyz_cyl[:,0].max()-250,n)
        ys = np.linspace(xyz_cyl[:,1].min()+250, xyz_cyl[:,1].max()-250,n)

        for ii in range(n):
            for jj in range(n):
                x = xs[ii]
                y = ys[jj]
                dr_los, dm_los, zlos = frb.compute_dm_los(xyz_cyl,[x,y,0],
                                                          DM,cellsize)
                sort_index = np.argsort(zlos)
                zlos = zlos[sort_index]
                dm_los = dm_los[sort_index]
                DMarr[ii,jj,kk] = np.sum(dm_los)/h**2

    return DMarr

def create_cube():
    fl = glob.glob('data_1e4kpc_100000-100000/xyz_cell_chun*.npy')
    datacube = []


def cgm_interveners(outdir='data_1e4kpc_100000-100000/', plot=False):
    outdir = 'data_1e4kpc_100000-100000/'
    outdir='data_25e3kpc_100000-100000/'
    halos = frb.read_groups(Mmin=1e12, Mmax=np.inf)
    xyz_halos = halos[0]
    Mhalo = halos[-2].value
    r500 = halos[-1]
    bperpmin = []
    dDM = []
    r500min = []
    Mhalos_x = []
    M = []
    for kk in range(24):
        for jj in range(24):
            xlos = [76000+jj*2e3,124000+kk*2e3,100000]
            print(kk,jj)

            if kk==0 and jj==0:
                xyz_cyl, zlos, dm_los, DM, cellsize = dm_from_cyl(outdir, xlos=xlos, 
                                                              nlosx=1, nlosy=1, 
                                                              dochunks=False)
            else:
                dr_los, dm_los, zlos = frb.compute_dm_los(xyz_cyl,xlos,DM,cellsize)
                sort_index = np.argsort(zlos)

                zlos = zlos[sort_index] 
                dm_los = dm_los[sort_index]

            if not len(dm_los):
                continue

            dist = np.sqrt(np.sum((xyz_halos[:, :2] - xlos[:2])**2, 1))
            distnorm = dist / r500
            ind = np.where(dist/r500 < 5.0)[0]
            if not len(ind):
                continue
            Mhalos_x.append(Mhalo[ind].max())
            M.append(Mhalo[ind])
            a, b, c = np.argmin(distnorm), np.min(dist), np.min(distnorm)
            bperpmin.append(b)
            r500min.append(c)
            dDM.append(np.sum(dm_los))

            if plot:
                figure()
                plot(1.4e-3* zlos, np.cumsum(dm_los), c='k', lw=3)
                xlabel('Distance (Mpc)')
                ylabel('Dispersion Measure')
                for ii in ind:
                    axvline(1.4e-3 * xyz_halos[ii, 2], color='C1', linestyle=':', alpha=0.5)
                    text(1.4e-3 * (xyz_halos[ii, 2] + 4e3), np.random.uniform(np.cumsum(dm_los).max()*0.1, np.cumsum(dm_los).max()*0.75),
                        '%0.1fe11 Msun\n b=%0.1f kpc'%(halos[-2][ii].value/1e11, dist[ii]))
                savefig('DM%d-%d.pdf' % (kk, jj))

    np.save('Mhalos_intervener', M)
    np.save('bperp_min_kpc', bperpmin)
    np.save('r500min', r500min)
    np.save('DMlos.npy', dDM)

def dm_from_halos():
    fl = glob.glob('/home/connor/TNG300-1/losdata2/dm*.npy')
    halos = frb.read_groups(Mmin=1e11, Mmax=np.inf)
    xyz_halos = halos[0]
    Mhalo = halos[-2].value
    r500 = halos[-1]
    bperpmin = []
    dDM = []
    r500min = []
    Mhalos_x = []
    M = []
    Mr500min = []
    Narr=[]
    B=[]
    ind=[]
    zarr, dmarr = [], []
    for fn in fl[:]:
        dm_los = np.load(fn)
        z = np.load(fn.replace('dm', 'z'))
        xx = int(fn.split('dm')[1].split('-')[0])
        yy = int(fn.split('dm')[1].split('-')[-1].split('.')[0])
        xlos = [xx,yy]

        dist = np.sqrt(np.sum((xyz_halos[:, :2] - xlos[:2])**2, 1))

#        if dist[Mhalo>1e13].min() < 2000:
#            print("Skipping because too close to group/cluster")
#            continue

        distnorm = dist / r500
        ind = np.where(distnorm < 5)[0]

        if np.sum(dm_los) > 500.:
            print(dist[ind], np.log10(Mhalo[ind]))
            plot(z, np.cumsum(dm_los))
            for nn in range(len(ind)):
                axvline(xyz_halos[nn][-1])
            break

        if not len(ind):
            plot(z, np.cumsum(dm_los))
            continue

        Narr.append(len(ind))
        dDM.append(np.sum(dm_los))
        indarr.append(ind)
        indo = np.argmin(dist/r500)
        Mhalos_x.append(Mhalo[indo])
        M.append(Mhalo[ind])
        B.append(dist[ind])
        a, b, c = np.argmin(distnorm), np.min(dist), np.min(distnorm)
        bperpmin.append(b)
        r500min.append(c)
        #dDM.append(np.sum(dm_los))

    dDM = np.array(dDM)
    r500min = np.array(r500min)
    Mhalos_x = np.array(Mhalos_x)
    bperpmin = np.array(bperpmin)

def get_dm_dist(fl, radius_cyl_kpc=5000):
    frb = IllustrisFRB("output/", 98, "basedir")
    halos = frb.read_groups(Mmin=5e9, Mmax=np.inf)
    xyz_halos = halos[0]
    Mhalo = halos[-2].value
    r500 = halos[-1]
    dmFM = np.zeros([len(fl)])
    Ngal_cyl, Mtot_cyl, dmarr, Ngal_x, Mmax_arr = [], [], [], [], []
    dm_model = []
    Bperpmin = []

    for jj,fn in enumerate(fl[:]):
        dm_los = np.load(fn)
        z = np.load(fn.replace('dm', 'z'))
        xx = int(fn.split('dm')[1].split('-')[0])
        yy = int(fn.split('dm')[1].split('-')[-1].split('.')[0])
        xlos = [xx, yy, 0]
        dist = np.sqrt(np.sum((xyz_halos[:, :2] - xlos[:2])**2, 1))
        distnorm = dist / r500

        indcyl = np.where(dist < radius_cyl_kpc)[0]
        indx = np.where((distnorm < 7) \
                        & (Mhalo > 1e11) \
                        & (Mhalo < 1e13))[0]

        if not len(indx):
            continue

        Ngal_cyl.append(len(indcyl))
        Mtot_cyl.append(np.sum(Mhalo[indcyl]))
        dmarr.append(np.sum(dm_los))
        Ngal_x.append(len(indx))
        Bperpmin.append(distnorm[indx].min())

        if len(indx)==0:
            Mmax_arr.append(0)
        else:
            Mmax_arr.append(np.max(Mhalo[indx]))

        dmFM = 0
        for ll in range(len(indx)):
            logM = np.log10(Mhalo[indx[ll]])
            if logM>13.:
                m = models.ICM(logM)
            else:
                m = models.M31(logM)
            dmii = m.Ne_Rperp(dist[indx[ll]]*u.kpc)
            dmFM += dmii.value
        dm_model.append(dmFM)

    Ndensity_gal = np.array(Ngal_cyl) / (np.pi * radius_cyl_kpc**2 * 300e3)

    return Ndensity_gal, Mtot_cyl, dmarr, Ngal_x, Mmax_arr, dm_model

def cgm_likelihood():
    nfrb = len(Mhalo)
    DM_cgm_ii = []
    Delta_DM = []
    for ii in range(nfrb):
        for jj in range(len(Mhalo[ii])):
            #halo_model = m.YF17(Mhalo[ii][jj])
            DM_cgm_ii.append(1)#halo_model.Ne_Rperp(bperp[ii][jj]*u.kpc).value)

        DM_ex_model = np.sum(DM_cgm_ii)
    
        Delta_DM.append(DM_ex_obs - DM_ex_model)

    sigma_DM = 10 * np.ones_like(Delta_DM)

    logL = np.sum(-0.5 * Delta_DM**2 / Delta)

    return logL



if __name__=='__main__':
    frb = IllustrisFRB("output/", 98, "basedir")