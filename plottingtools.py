import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

def dm_animation(xyz_cyl, zlos, dm_los, 
                 nzarr=5245, step=4, 
                 save=False,
                 ythresh=np.inf):
    """ Make an animation or save down a large number of pngs
    """
    from celluloid import Camera
    indz = np.where(np.abs(xyz_cyl[:,1]-np.median(xyz_cyl[:,1]))<ythresh)[0]
    arr = np.histogram2d(xyz_cyl[indz,0], xyz_cyl[indz,2], bins=(300, nzarr))[0]
    x = np.linspace(0,0.5,5245)
    zarr = np.linspace(zlos[0],zlos[-1],nzarr)

    f = interpolate.interp1d(zlos, np.cumsum(dm_los))
    zlos_full = np.linspace(zlos.min(), zlos.max(), nzarr)
    dm_los_full = f(zlos_full)

    fig, axes = plt.subplots(1,2,figsize=(11.5,5))

    # if not save:
    #     fig, axes = plt.subplots(1,2,figsize=(11.5,5))
    # else:
    #     import matplotlib as mpl
    #     mpl.use('Agg')

    camera = Camera(fig)
    nstep = 5245//step-256

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
                axes[0].plot(x[x<0.45], y[x<0.45], ':', c='white', alpha=1.0, lw=1)

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

def plot_halos_paper(cmap='plasma',sep_thresh=2500):
    frb = IllustrisFRB("output/", 98, "basedir")
    #halos = frb.read_halos(Mmin=4.8e14, Mmax=4.85e14)
    #xyz_cyl, zlos, dm_los = dm_from_cyl(halos[0][0]+np.array([300,300,0]))
    
    halos = frb.read_halos(Mmin=1e14, Mmax=np.inf)

    arr_list = []
    dm_of_b_list=[]

    for ii in range(4):
        xyz_halo = halos[0][ii]
        Mhalo = halos[4][ii]

        fn = 'data_%d-%d/xyz_cyl.npy' % (int(xyz_halo[0]),int(xyz_halo[1]))
        xyz, zlos, dm_los, DM, cellsize = dm_from_cyl(halos[0][ii], 
                                            outdir=None, 
                                            sav=False)

        sep_kpc = np.sqrt(np.sum(np.abs(xyz_halo-xyz)**2, axis=1))
        indz = np.where(sep_kpc<sep_thresh)[0]
        dm_arr = DM[indz]

        rs, dm_of_b = get_dm_profile(xyz[indz], xyz_halo, dm_arr, 
                                    cellsize[indz], nr=50,ntheta=20)
        dm_of_b_list.append(dm_of_b)


        arr = plt.hist2d(xyz[indz,0], xyz[indz,1], 
                         bins=512, 
                         cmap='afmhot',
                         weights=dm_arr)[0]

        arr_list.append(arr)

    fig = plt.figure(figsize=(10,5))
    extent = [-sep_thresh, sep_thresh, -sep_thresh, sep_thresh]

    gs = GridSpec(3, 4, figure=fig, hspace=0.0, wspace=0.0)

    ax1 = fig.add_subplot(gs[:2,0])
    ax1.imshow(np.log10(arr_list[0]), 
               interpolation='nearest', 
               cmap=cmap,
               extent=extent)

    ax1.set_xticks([])
    ax1.set_ylabel('Mpc', fontsize=15)

    ax2 = fig.add_subplot(gs[2,0])
    ax2.set_ylabel('DM')
    ax2.plot(rs, dm_of_b_list[0].T)

    ax3 = fig.add_subplot(gs[:2,1])
    ax3.set_xticks([])
    ax3.set_yticks([])
    ax3.imshow(np.log10(arr_list[1]), 
               interpolation='nearest', 
               extent=extent,
               cmap=cmap)

    ax4 = fig.add_subplot(gs[2,1])
    ax4.set_yticks([])
    ax4.plot(rs, dm_of_b_list[1].T)


    ax5 = fig.add_subplot(gs[:2,2])
    ax5.set_xticks([])
    ax5.set_yticks([])
    ax5.imshow(np.log10(arr_list[2]), 
               interpolation='nearest', 
               extent=extent,
               cmap=cmap)

    ax6 = fig.add_subplot(gs[2,2])
    ax6.set_yticks([])
    ax6.plot(rs, dm_of_b_list[2].T)


    ax7 = fig.add_subplot(gs[:2,3])
    ax7.set_xticks([])
    ax7.set_yticks([])
    ax7.imshow(np.log10(arr_list[3]), 
               interpolation='nearest',
               extent=extent, 
               cmap=cmap)

    ax8 = fig.add_subplot(gs[2,3])
    ax8.set_yticks([])
    ax8.plot(rs, dm_of_b_list[3].T)