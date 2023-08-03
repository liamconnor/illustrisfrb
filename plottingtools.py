import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import illustris_python as il 

from illustrisfrb import IllustrisFRB

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

colors = ['k', '#482677FF', '#238A8DDF', '#95D840FF']

def compare_fwd_model(dmTNG, dmFM, maxM):
    """ Compare the forward modelled FRB 
        excess DMs to the TNG DM
        on a per-sightline basis"""

    bperp_arr = np.load('pickledB.npy.npz')
    Mhalo_arr = np.load('pickledM.npy.npz')
    dmTNG = np.load('pickledDM.npy')
    maxM, dmFM = np.zeros([len(dmTNG)]), np.zeros([len(dmTNG)])

    for ii in range(len(dmTNG)):
        bperp = bperp_arr['arr_%d' % ii]
        Mhalo = Mhalo_arr['arr_%d' % ii]
        print(np.log10(Mhalo.min()), np.log10(Mhalo.max()))
        for jj in range(len(Mhalo)):
            logM = np.log10(M['arr_%d' % ii][jj])
            if logM>13.:
                m = models.ICM(logM)
            else:
                m = models.M31(logM, alpha=2.5)
            dmii = m.Ne_Rperp(bperp[jj]*u.kpc)
            dmFM[ii] += 1.17*dmii.value
            maxM[ii] = np.max(np.log10(M['arr_%d' % ii]))

    fig = plt.figure(figsize=(8,5))
    x = np.logspace(-2,5,50)
    plt.fill_between(x, x, 10000*x, alpha=0.1)
    plt.fill_between(x, 0, x, alpha=0.1, color='r')
    plt.text(10, 1000, 'Forward model overpredicts')
    plt.text(150, 1, 'IllustrisTNG overpredicts')
    plt.scatter(dmTNG, dmFM, c=maxM, alpha=0.85, s=10, )
    plt.colorbar(label=r'Max halo mass (M$_\odot$)')
    plt.xlabel(r'Illustris DM (pc cm$^{-3}$)', fontsize=15)
    plt.ylabel(r'Fwd modelled DM (pc cm$^{-3}$)', fontsize=15)
    plt.xlim(0.5, 5000)
    plt.ylim(0.5, 3e3)
    plt.loglog()
    plt.show()

def plot_sightline(fl, r500_min=1, Mhalo_min=1e9, Mhalo_max=np.inf):
    frb = IllustrisFRB("output/", 98, "basedir")
    halos = frb.read_groups(Mmin=5e9, Mmax=np.inf)
    xyz_halos = halos[0]
    Mhalo = halos[-2].value
    r500 = halos[-1]
    dmFM = np.zeros([len(fl)])
    Ngal_cyl, Mtot_cyl, dmarr, Ngal_x, Mmax_arr = [], [], [], [], []
    dm_model = []
    for jj,fn in enumerate(fl[:]):
        dm_los = np.load(fn)
        z = np.load(fn.replace('dm', 'z'))
        xx = int(fn.split('dm')[1].split('-')[0])
        yy = int(fn.split('dm')[1].split('-')[-1].split('.')[0])
        xlos = [xx, yy, 0]
        dist = np.sqrt(np.sum((xyz_halos[:, :2] - xlos[:2])**2, 1))
        distnorm = dist / r500

        indcyl = np.where(dist < 5000.)[0]
        indx = np.where((distnorm < r500_min) & (Mhalo > Mhalo_min) & (Mhalo < Mhalo_max))[0]

        Ngal_cyl.append(len(indcyl))
        Mtot_cyl.append(np.sum(Mhalo[indcyl]))
        dmarr.append(np.sum(dm_los))
        Ngal_x.append(len(indx))

        if len(indx)==0:
            continue
            Mmax_arr.append(0)
        else:
            Mmax_arr.append(np.max(Mhalo[indx]))

        # dmFM = 0
        # for ll in range(len(indx)):
        #     logM = np.log10(Mhalo[indx[ll]])
        #     if logM>13.:
        #         m = models.ICM(logM)
        #     else:
        #         m = models.M31(logM)
        #     dmii = m.Ne_Rperp(dist[indx[ll]]*u.kpc)
        #     dmFM += dmii.value
        # dm_model.append(dmFM)

        # if np.min(dist[indx]) > 75.:
        #     continue

        colors = plt.cm.Dark2(range(8))
        fig = plt.figure(figsize=(8,5))
        plt.subplot(311)
        plt.scatter(xyz_halos[indcyl][:, 2], xyz_halos[indcyl][:, 0], 
                    s=Mhalo[indcyl]/1e11, c=colors[0])
        plt.axhline(xx, linestyle=':', c='k')
        plt.xlabel('z (cMpc)')
        plt.ylabel('x (cMpc)')

        plt.subplot(312)
        plt.scatter(xyz_halos[indcyl][:, 2], xyz_halos[indcyl][:, 1], 
                    s=Mhalo[indcyl]/1e11, c=colors[1])
        plt.axhline(yy, linestyle=':', c='k')
        plt.xlabel('z (cMpc)')
        plt.ylabel('y (cMpc)')

        plt.subplot(313)
        plt.plot(z, np.cumsum(dm_los), c='k')
        plt.xlabel('z (cMpc)')
        plt.ylabel('DM (pc cm$^{-3}$)')
        
        for kk,ii in enumerate(indx):
            plt.axvline(xyz_halos[ii][-1], c='C1')
            if np.log10(Mhalo[ii]) > 13.:
                m = models.ICM(np.log10(Mhalo[ii]))
            else:
                m = models.M31(np.log10(Mhalo[ii]))
            dmii = m.Ne_Rperp(dist[ii]*u.kpc)
            dmFM[jj] += dmii.value
            plt.text(xyz_halos[ii][-1], 
                    kk/len(indx)*(np.sum(dm_los)), 
                    'logM=%0.1f\nBperp=%d kpc\nDM$_{mod}$=%0.1f' \
                    % (np.log10(Mhalo[ii]),dist[ii],dmii.value),
                    )
        plt.tight_layout()
        #plt.savefig('plots/%d-%d.png' % (xx, yy))
            

def dm_animation(xyz_cyl, zlos, dm_los, 
                 nzarr=5245, step=4, 
                 save=False,
                 ythresh=np.inf):
    """ Make an animation or save down a large number of pngs
    """
    from celluloid import Camera
    indz = np.where(np.abs(xyz_cyl[:,1]-np.median(xyz_cyl[:,1]))<ythresh)[0]
    arr = np.histogram2d(xyz_cyl[indz,0], xyz_cyl[indz,2], 
                         bins=(300, nzarr))[0]
    x = np.linspace(0,0.5,5245)
    zarr = np.linspace(zlos[0],zlos[-1],nzarr)

    f = interpolate.interp1d(zlos, np.cumsum(dm_los))
    zlos_full = np.linspace(zlos.min(), zlos.max(), nzarr)
    dm_los_full = f(zlos_full)

    fig, axes = plt.subplots(1,2,figsize=(11.5,5))

    if not save:
        fig, axes = plt.subplots(1,2,figsize=(11.5,5))
    else:
        import matplotlib as mpl
        mpl.use('Agg')

    camera = Camera(fig)2

    for i in range(nstep)[:]:
        if save:
            fig, axes = plt.subplots(1,2,figsize=(10,5))
        
        arr_snap = arr[22:-22,step*i-128:step*i-128+256]
        
        ti = int(step*i+1)
        dm_snap = dm_los_full[:ti]
        zlos_snap = zlos_full[:ti]

#        if len(arr_snap[-1])==0:
#            continue

        axes[0].imshow(np.log10(arr_snap+1), 
                       cmap='afmhot', 
                       aspect='auto',
#                       extent=[zarr[4*i],zarr[4*i+256],0,1], 
                       extent=[0,1,0,1],   
                       vmax=np.log10(arr.max())*0.85, 
                       vmin=0.0,)

        axes[0].axis('off')


#        if step*i<256:
        if i<0:
            xx = np.linspace(0, 0.5*i*step/256., 5245)
            tdelay1 = dm_snap[-1]/dm_los_full[-1]*0.1 + 0.02
            g1 = gauss(xx, (0.5*i*step/256.-0.02-tdelay1), 0.0175)
            g3 = gauss(xx, (0.5*i*step/256.-0.02-tdelay), 0.0175)
            g2 = gauss(xx, (0.5*i*step/256.-0.02), 0.0175)
            y = 0.5 + 0.08*np.sin(1.75 * np.pi * (40*xx + 0.02 * i)) * g1 
            axes[0].plot(xx, y, c='red', alpha=0.9, lw=2)
            y = 0.5 + 0.08*np.sin(1.75 * np.pi * (60*xx + 0.02 * i)) * g2
            axes[0].plot(xx, y, c='C0', alpha=0.8, lw=2)
        else:
            if i < -1:
                c1 = 'white'
                tdelay1 = dm_snap[-1]/dm_los_full[-1]*0.1 + 0.02
                xx = np.linspace(0, 0.5*i*step/256., 5245)
                g1 = gauss(xx, (0.5*i*step/256.-0.02-tdelay1), 0.0175)
                g3 = gauss(xx, (0.5*i*step/256.-0.02-tdelay2), 0.0175)
                g2 = gauss(xx, (0.5*i*step/256.-0.02), 0.0175)
                y = 0.5 + 0.08*np.sin(1.75 * np.pi * (60*x + 0.02 * i)) * g2
                axes[0].plot(x[x>(0.48-tdelay1-0.05)], y[x>(0.48-tdelay1-0.05)], 
                             c='red', alpha=0.5, lw=1)
                axes[0].plot(x[x>(0.48-tdelay3-0.05)], y[x>(0.48-tdelay3-0.05)], 
                             c='green', alpha=0.5, lw=1)
                axes[0].plot(x[x>0.45], y[x>0.45], c=c1, alpha=1.0, lw=1)
                axes[0].plot(x[x<0.45], y[x<0.45], ':', c=c1, alpha=0.75, lw=2)
            else:
                tdelay1 = dm_snap[-1]/dm_los_full[-1]*0.1 + 0.02
                tdelay3 = dm_snap[-1]/dm_los_full[-1]*0.05 + 0.01
                c1 = 'C0'
                c1 = 'white'
                g1 = gauss(x, 0.48-tdelay1, 0.0175)
                g3 = gauss(x, 0.48-tdelay3, 0.0175)
                g2 = gauss(x, 0.48, 0.0175)

                y = 0.5 + 0.08*np.sin(1.75 * np.pi * (40*x + 0.02 * i)) * g1 
                #axes[0].plot(x[x>(0.48-tdelay1-0.05)], y[x>(0.48-tdelay1-0.05)], 
                #             c='red', alpha=1.0, lw=1)

                y = 0.5 + 0.08*np.sin(1.75 * np.pi * (60*x + 0.02 * i)) * g3
                #axes[0].plot(x[x>(0.48-tdelay3-0.05)], y[x>(0.48-tdelay3-0.05)], 
                #             c='green', alpha=0.9, lw=1)

                y = 0.5 + 0.08*np.sin(1.75 * np.pi * (60*x + 0.02 * i)) * g2
                axes[0].plot(x[x>0.45], y[x>0.45], c=c1, alpha=0.25, lw=2.5)
                axes[0].plot(x[x>0.45], y[x>0.45], c=c1, alpha=1.0, lw=1.0)
                axes[0].plot(x[x<0.45], y[x<0.45], ':', c='white', alpha=1., lw=1.5)

        axes[1].plot(zlos_snap,dm_snap,lw=2.5,c='k',)
        axes[1].set_xlim(0,max(10., zlos_snap.max()*1.5))
        axes[1].set_xlabel('Distance (cMpc)', fontsize=15)
        axes[1].set_ylabel('DM (pc / cm**3)', fontsize=15)
        axes[1].set_xlim(0,max(10., zlos_snap.max()*1.5))

        if save==True:
            plt.savefig('plots/%d'%i)

        camera.snap()
    animation = camera.animate(repeat=True, interval=10)
#    animation.save('celluloid_minimal.gif', writer = 'imagemagick')

def gauss(x,mu,sig):
    return np.exp(-(x-mu)**2/sig**2)

def dm_from_cyl_paper(outdir=None, sav=True, xlos=None):
    FRBIl = IllustrisFRB("output/", 98, "basedir")

    if outdir is None:
        outdir = './data_full%d-%d'%(xlos[0], xlos[1])

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

def plot_halos_paper(cmap='plasma',sep_thresh=2500):
    frb = IllustrisFRB("output/", 98, "basedir")
    #halos = frb.read_halos(Mmin=4.8e14, Mmax=4.85e14)
    #xyz_cyl, zlos, dm_los = dm_from_cyl(halos[0][0]+np.array([300,300,0]))
    
    halos = frb.read_groups(Mmin=1e14, Mmax=np.inf)

    nr, ntheta = 30,75
    arr_list = []
    dm_of_b_arr = np.zeros([4, ntheta, nr])
    mhalo_arr = []
    kk = 0

    for ii in range(30,34):
        xyz_halo = halos[0][ii]
        Mhalo = halos[4][ii]
        mhalo_arr.append(Mhalo.value)

        fn = 'data_%d-%d/xyz_cyl.npy' % (int(xyz_halo[0]),int(xyz_halo[1]))
        xyz, zlos, dm_los, DM, cellsize = dm_from_cyl_paper(xlos=halos[0][ii], 
                                                            outdir=None, 
                                                            sav=False)

        sep_kpc = np.sqrt(np.sum(np.abs(xyz_halo-xyz)**2, axis=1))
        indz = np.where(sep_kpc<sep_thresh)[0]
        dm_arr = DM[indz]

        rs, dm_of_b = get_dm_profile(xyz[indz], xyz_halo, dm_arr, 
                                    cellsize[indz], 
                                    nr=nr,
                                    ntheta=ntheta)
        dm_of_b_arr[kk] = dm_of_b

        arr = plt.hist2d(xyz[indz,0], xyz[indz,1], 
                         bins=512, 
                         cmap='afmhot',
                         weights=dm_arr)[0]

        arr_list.append(arr)
        kk += 1

    np.save('Mhalo_plot', mhalo_arr)
    np.save('gas_array_plot', arr_list)
    np.save('dm_profile_plot',dm_of_b_arr)

    mhalo_arr = np.load('Mhalo_plot.npy')
    arr_list = np.load('gas_array_plot.npy')
    dm_of_b_arr = np.load('dm_profile_plot.npy')*0.5
    rs = np.load('rs_plot.npy')

    dm_of_b_arr *= 1.1666

    sep_thresh = 2000.
    ymax = 4500
    fig = plt.figure(figsize=(10,5))
    extent = [-sep_thresh, sep_thresh, -sep_thresh, sep_thresh]


    bperp_elektra, dm_elektra_mean = 520., 416.
    bperp_jackie, dm_jackie_mean = 850., 93.
    yerr_elektra = np.array([151, 92]).reshape(2,1)
    yerr_jackie = np.array([54, 82.7]).reshape(2,1)

    gs = GridSpec(3, 4, figure=fig, hspace=0.0, wspace=0.0)

    ax1 = fig.add_subplot(gs[:2,0])
    ax1.imshow(np.log10(arr_list[0]), 
               interpolation='nearest', 
               cmap=cmap,
               extent=extent)
    circle_elektra = plt.Circle((0, 0), 450, edgecolor=colors[2], fill=False,linestyle=':')
    circle_jackie = plt.Circle((0, 0), 950, edgecolor=colors[3], fill=False,linestyle=':')

    ax1.set_xticks([])
    ax1.set_ylabel('kpc', fontsize=14)
    ax1.set_title(r'$M_{500}=%0.2f\times10^{14}M_{\odot}$'%(mhalo_arr[0]/1e14), fontsize=12)
    ax1.add_patch(circle_elektra)
    ax1.add_patch(circle_jackie)

    ax2 = fig.add_subplot(gs[2,0])
    ax2.set_ylabel(r'DM (pc cm$^{-3}$)')
    sig = np.std(dm_of_b_arr[0], axis=0)
    med = np.median(dm_of_b_arr[0], axis=0)

    ax2.fill_between(rs, med-1.6*sig, 
                         med+1.6*sig, alpha=0.25, color=colors[1])   
    ax2.plot(rs, med, c=colors[1])
    ax2.set_xlabel(r'$b_{\perp}$ (kpc)')
    ax2.set_ylim(10,ymax)
    ax2.hlines(450, 0, 1)
    ax2.semilogy()
    ax2.errorbar(bperp_elektra, dm_elektra_mean, yerr=yerr_elektra, 
                marker='o', lw=2, markersize=3, barsabove=True, capsize=4, color=colors[2])
    ax2.errorbar(bperp_jackie, dm_jackie_mean, 
                yerr=yerr_jackie, marker='o', lw=2, markersize=3, 
                barsabove=True, capsize=4, color='limegreen')
    ax2.set_xticks([-2000, -1000, 1000])

    ax3 = fig.add_subplot(gs[:2,1])
    circle_elektra = plt.Circle((0, 0), 450, edgecolor=colors[2], fill=False,linestyle=':')
    circle_jackie = plt.Circle((0, 0), 950, edgecolor=colors[3], fill=False,linestyle=':')
    ax3.set_xticks([])
    ax3.set_yticks([])
    ax3.set_title(r'$M_{500}=%0.2f\times10^{14}M_{\odot}$'%(mhalo_arr[1]/1e14), fontsize=12)
    ax3.imshow(np.log10(arr_list[1]), 
               interpolation='nearest', 
               extent=extent,
               cmap=cmap)
    ax3.add_patch(circle_elektra)
    ax3.add_patch(circle_jackie)

    ax4 = fig.add_subplot(gs[2,1])
#    ax4.set_yticks([])
    sig = np.std(dm_of_b_arr[1], axis=0)
    med = np.median(dm_of_b_arr[1], axis=0)

    ax4.fill_between(rs, med-1.6*sig, 
                         med+1.6*sig, alpha=0.25, color=colors[1])    
    ax4.plot(rs, med, c=colors[1])
    ax4.set_xlabel(r'$b_{\perp}$ (kpc)')
    ax4.set_ylim(10,ymax)
    ax4.semilogy()
    ax4.errorbar(bperp_elektra, dm_elektra_mean, yerr=yerr_elektra, 
                marker='o', lw=2, markersize=3, barsabove=True, capsize=4, color=colors[2])
    ax4.errorbar(bperp_jackie, dm_jackie_mean, 
                yerr=yerr_jackie, marker='o', lw=2, markersize=3, 
                barsabove=True, capsize=4, color='limegreen')
    ax4.set_xticks([-1000, 1000, 2000])

    ax5 = fig.add_subplot(gs[:2,2])
    circle_elektra = plt.Circle((0, 0), 450, edgecolor=colors[2], 
                        fill=False,linestyle=':')
    circle_jackie = plt.Circle((0, 0), 950, edgecolor=colors[3], 
                        fill=False,linestyle=':')
    ax5.set_xticks([])
    ax5.set_yticks([])
    ax5.set_title(r'$M_{500}=%0.2f\times10^{14}M_{\odot}$'%(mhalo_arr[2]/1e14), fontsize=12)

    ax5.imshow(np.log10(arr_list[2]), 
               interpolation='nearest', 
               extent=extent,
               cmap=cmap)
    ax5.add_patch(circle_elektra)
    ax5.add_patch(circle_jackie)

    ax6 = fig.add_subplot(gs[2,2])
#    ax6.set_yticks([])
    sig = np.std(dm_of_b_arr[2], axis=0)
    med = np.median(dm_of_b_arr[2], axis=0)

    ax6.fill_between(rs, med-1.6*sig, 
                         med+1.6*sig, alpha=0.25, color=colors[1])    
    ax6.plot(rs,med, c=colors[1])
    ax6.set_xlabel(r'$b_{\perp}$ (kpc)')
    ax6.set_ylim(10,ymax)
    ax6.semilogy()
    ax6.errorbar(bperp_elektra, dm_elektra_mean, yerr=yerr_elektra, 
                marker='o', lw=2, markersize=3, barsabove=True, capsize=4, color=colors[2])
    ax6.errorbar(bperp_jackie, dm_jackie_mean, 
                yerr=yerr_jackie, marker='o', lw=2, markersize=3, 
                barsabove=True, capsize=4, color='limegreen')
    ax6.set_xticks([-1000, 1000])

    ax7 = fig.add_subplot(gs[:2,3])
    circle_elektra = plt.Circle((0, 0), 450, edgecolor=colors[2], fill=False,linestyle=':')
    circle_jackie = plt.Circle((0, 0), 950, edgecolor=colors[3], fill=False,linestyle=':')
    ax7.set_xticks([])
    ax7.set_yticks([])
    ax7.imshow(np.log10(arr_list[3]), 
               interpolation='nearest',
               extent=extent, 
               cmap=cmap)
    ax7.set_title(r'$M_{500}=%0.2f\times10^{14}M_{\odot}$'%(mhalo_arr[3]/1e14), fontsize=12)
    ax7.add_patch(circle_elektra)
    ax7.add_patch(circle_jackie)

    ax8 = fig.add_subplot(gs[2,3])
    sig = np.std(dm_of_b_arr[3], axis=0)
    med = np.median(dm_of_b_arr[3], axis=0)

    ax8.fill_between(rs, med-1.6*sig, 
                         med+1.6*sig, alpha=0.25, color=colors[1])    
    ax8.plot(rs,med, c=colors[1])
    ax8.set_xlabel(r'$b_{\perp}$ (kpc)')
    ax8.semilogy()
    ax8.set_ylim(10,ymax)
    ax8.errorbar(bperp_elektra, dm_elektra_mean, yerr=yerr_elektra, 
                marker='o', lw=2, markersize=3, barsabove=True, 
                capsize=4, color=colors[2])
    ax8.errorbar(bperp_jackie, dm_jackie_mean, 
                yerr=yerr_jackie, marker='o', lw=2, markersize=3,
                barsabove=True, capsize=4, color='limegreen', )
    ax8.set_xticks([-1000, -500, 500, 1000])
#    plt.suptitle('IllustrisTNG Clusters')

    ax2.set_xticks([-1000, 0, 1000])
    ax4.set_xticks([-1000, 0, 1000])
    ax6.set_xticks([-1000, 0, 1000])
    ax8.set_xticks([-1000, 0, 1000])

    ax4.set_yticks([])
    ax6.set_yticks([])
    ax8.set_yticks([])
    plt.savefig('./cluster_sim_fig.pdf')



