import matplotlib.pyplot as plt
import numpy as np

def plot_z_dist(catalog,
                bins=np.linspace(0,3,301),
                log=False,
                path='',
                color='#377eb8',  
                title=r'$Z_{true}$ Histogram'):
    
    plt.hist(catalog['redshift'], bins=bins, log=log, color=color, edgecolor='none', alpha=0.8)
    plt.title(title, fontsize=16)
    plt.xlabel('Redshift', fontsize=14)
    plt.ylabel('Count', fontsize=14)
    plt.grid(True, which='both', linestyle='--', alpha=0.6)
    if path:
        plt.savefig(path)
    plt.show()
    

def plot_mag_color(catalog,
              title='',
              save=False,
              path='',
              figsize=(9, 13),
              xlim=[16, 32],
              ylim=[-2, 5],
              cmap='Reds',
              gridsize=[400, 200],
              bins='log'):
    """
    Plots color-magnitude diagrams for different bands from a catalog.

    Parameters:
    catalog (pd.DataFrame): DataFrame containing columns 'mag_band_lsst' for u, g, r, i, z, y.
    title (str, optional): Title of the plot. Defaults to ''.
    save (bool, optional): Whether to save the plot. Defaults to False.
    path (str, optional): Path where to save the plot if save=True. Defaults to ''.
    figsize (tuple, optional): Figure size. Defaults to (9, 13).
    xlim (list, optional): Limits for the x-axis. Defaults to [16, 32].
    ylim (list, optional): Limits for the y-axis. Defaults to [-2, 5].
    cmap (str, optional): Colormap. Defaults to 'Reds'.
    gridsize (list, optional): Grid size for hexbin plot. Defaults to [400, 200].
    bins (str or int, optional): Binning strategy for hexbin plot. Defaults to 'log'.

    Returns:
    None
    """
    bands = ['u', 'g', 'r', 'i', 'z', 'y']
    plt.figure(figsize=figsize)
    i = 1
    
    for band, _band in zip(bands, bands[1::]):
        plt.subplot(3, 2, i)
        i += 1
        
        mag_diff_v = catalog[f'mag_{band}_lsst'] - catalog[f'mag_{_band}_lsst']
        mag_v = catalog[f'mag_{band}_lsst']
                   
        plt.hexbin(mag_v, mag_diff_v, None, mincnt=1, cmap=cmap, gridsize=gridsize, bins=bins)
        
        plt.xlabel(f"mag {band}", fontsize=13)
        plt.ylabel(f"{band}-{_band}", fontsize=13)
        plt.xlim(xlim)
        plt.ylim(ylim)
        plt.suptitle(title, fontsize=16)
        plt.tight_layout()
    
    if save:
        plt.savefig(path)
        print(f"Plot saved as {path}")
    else:
        plt.show()
        
    
def plot_color_color(catalog,
                title='',
                save=False,
                path='',
                figsize=(12, 12),
                cmap='turbo',
                gridsize=[400, 400],
                bins='log',
                xlim=[-2,5],
                ylim=[-2,5]):
    """
    Plots color-color diagrams for different bands from a catalog.

    Parameters:
    catalog (pd.DataFrame): DataFrame containing columns 'mag_band_lsst' for u, g, r, i, z, y.
    title (str, optional): Title of the plot. Defaults to ''.
    save (bool, optional): Whether to save the plot. Defaults to False.
    path (str, optional): Path where to save the plot if save=True. Defaults to ''.
    figsize (tuple, optional): Figure size. Defaults to (12, 12).
    cmap (str, optional): Colormap. Defaults to 'turbo'.
    gridsize (list, optional): Grid size for hexbin plot. Defaults to [400, 400].
    bins (str or int, optional): Binning strategy for hexbin plot. Defaults to 'log'.

    Returns:
    None
    """
    bands = ['u', 'g', 'r', 'i', 'z', 'y']
    plt.figure(figsize=figsize)
    i = 1
    
    for index in range(len(bands) - 2):
        plt.subplot(3, 2, i)
        i += 1
        
        color = catalog[f'mag_{bands[index + 1]}_lsst']
        next_color = catalog[f'mag_{bands[index + 2]}_lsst']
        past_color = catalog[f'mag_{bands[index]}_lsst']
        
        plt.hexbin(past_color - color, color - next_color, None, mincnt=1, cmap=cmap, gridsize=gridsize, bins=bins)
        plt.ylabel(f'{bands[index + 1]}-{bands[index + 2]}', fontsize=13)
        plt.xlabel(f'{bands[index]}-{bands[index + 1]}', fontsize=13)
        cbar = plt.colorbar()
        plt.xlim(xlim)
        plt.ylim(ylim)
        plt.suptitle(title, fontsize=16)
        plt.tight_layout()
    
    if save:
        plt.savefig(path)
        print(f"Plot saved as {path}")
    else:
        plt.show()


def plot_density_sky(catalog):
    all_pix = set(catalog['pix'])
    density = {pix: len(catalog[catalog['pix'] == pix]) for pix in all_pix}
    total_objects = len(catalog)
    plt.figure(figsize=(10, 6))
    cmap = plt.cm.RdYlBu
    norm = plt.Normalize(vmin=min(density.values()), vmax=max(density.values()))
    for pix in all_pix:
        df = catalog[catalog['pix'] == pix]
        if not df.empty:
            plt.scatter(df['ra'], df['dec'], c=[density[pix]] * len(df), cmap=cmap, norm=norm, s=10, alpha=0.6)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar()
    cbar.set_label('Density of Objects in Pixels')
    plt.xlabel('Right Ascension (degrees)', fontsize=14)
    plt.ylabel('Declination (degrees)', fontsize=14)
    plt.title('Density of Objects in the Sky', fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()

def plot_max_band_magnitude(catalog, band):
    all_pix = set(catalog['pix'])
    all_max = [max(catalog[catalog['pix'] == pix][f'mag_{band}_lsst']) for pix in all_pix]
    plt.figure(figsize=(10, 6))
    cmap = plt.cm.viridis
    norm = plt.Normalize(vmin=min(all_max), vmax=max(all_max))
    for i, pix in enumerate(all_pix):
        df = catalog[catalog['pix'] == pix]
        plt.scatter(df['ra'], df['dec'], s=0.1, color=cmap(norm(all_max[i])), alpha=0.6)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar()
    cbar.set_label(f'Max depth {band}-band Magnitude')
    plt.xlabel('Right Ascension (degrees)', fontsize=14)
    plt.ylabel('Declination (degrees)', fontsize=14)
    plt.title(f'Max {band}-band Magnitude for Pixels', fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()


def plot_average_band_magnitude(catalog, band):
    all_pix = set(catalog['pix'])
    all_mean = [np.mean(catalog[catalog['pix'] == pix][f'mag_{band}_lsst']) for pix in all_pix]
    plt.figure(figsize=(10, 6))
    cmap = plt.cm.plasma
    norm = plt.Normalize(vmin=min(all_mean), vmax=max(all_mean))
    for i, pix in enumerate(all_pix):
        df = catalog[catalog['pix'] == pix]
        plt.scatter(df['ra'], df['dec'], s=0.1, color=cmap(norm(all_mean[i])), alpha=0.6)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar()
    cbar.set_label(f'Average {band}-band Magnitude')
    plt.xlabel('Right Ascension (degrees)', fontsize=14)
    plt.ylabel('Declination (degrees)', fontsize=14)
    plt.title(f'Average {band}-band Magnitude for Pixels', fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()


def plot_position(catalog, xlim=None, ylim=None, label='', color='#ff7f00', s=0.1, title='Sky Position', save=False, path=''):
    plt.figure(figsize=(10, 6))
    plt.scatter(catalog['ra'], catalog['dec'], s=s, color=color, label=label, alpha=0.6)
    plt.legend(fontsize=14, loc='upper left')
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.title(title, fontsize=16)
    plt.xlabel('Right Ascension (RA)', fontsize=14)
    plt.ylabel('Declination (Dec)', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.6)
    if save and path:
        plt.savefig(path)
    plt.show()

def plot_errors(catalog, title='Errors', gridsize=[400, 200], bins='log', cmap='inferno', xlim=[20, 30], ylim=[0, 100], save=False, path=''):
    bands = ['u', 'g', 'r', 'i', 'z', 'y']
    fig, axes = plt.subplots(3, 2, figsize=[12, 16])
    fig.suptitle(title, fontsize=16)
    for ax, band in zip(axes.flatten(), bands):
        mag = np.array(catalog[f'mag_{band}_lsst'])
        err = np.array(catalog[f'mag_err_{band}_lsst'])
        sn = 1 / (10 ** (0.4 * err) - 1)
        ax.hexbin(mag, sn, gridsize=gridsize, cmap=cmap, bins=bins, mincnt=1)
        ax.set_ylabel("S/N", fontsize=14)
        ax.set_xlabel(f"mag {band}", fontsize=14)
        ax.set_ylim(ylim)
        ax.set_xlim(xlim)
        ax.axhline(5, color='black', label='5σ', linestyle='--')
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend(fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    if save and path:
        plt.savefig(path)
    plt.show()

def plot_color_color_red(catalog, xlim=[-2, 5], ylim=[-2, 5], title='Color-Color Diagram with Redshift', bins=[900, 900], cmap='turbo', save=False, path=''):
    bands = ['u', 'g', 'r', 'i', 'z', 'y']
    fig = plt.figure(figsize=(12, 12))
    fig.suptitle(title, fontsize=16)
    for index in range(len(bands) - 2):
        plt.subplot(3, 2, index + 1)
        color = catalog[f'mag_{bands[index + 1]}_lsst']
        next_color = catalog[f'mag_{bands[index + 2]}_lsst']
        past_color = catalog[f'mag_{bands[index]}_lsst']
        plt.hexbin(past_color - color, color - next_color, C=catalog['redshift'], mincnt=1, cmap=cmap, gridsize=bins)
        plt.ylabel(f'{bands[index + 1]}-{bands[index + 2]}', fontsize=13)
        plt.xlabel(f'{bands[index]}-{bands[index + 1]}', fontsize=13)
        plt.colorbar(label='redshift')
        plt.xlim(xlim)
        plt.ylim(ylim)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    if save and path:
        plt.savefig(path)
    plt.show()

def plot_mag_histogram(catalog, bands=['u', 'g', 'r', 'i', 'z', 'y'], title='Magnitude Histograms', xlim=(15,35), save=False, path=''):
    plt.figure(figsize=(9, 13))
    bins = np.linspace(9, 37, 30)
    colors = ['#377eb8', '#4daf4a', '#e41a1c', '#ff7f00', '#984ea3', '#a65628']
    for i, band in enumerate(bands):
        plt.subplot(3, 2, i + 1)
        plt.hist(catalog[f'mag_{band}_lsst'], bins=bins, alpha=0.8, color=colors[i],log=True)
        plt.xlabel(f'{band} (mag)', fontsize=12)
        plt.ylabel('Number of galaxies', fontsize=12)
        plt.xlim(xlim)
        plt.grid(True, linestyle='--', alpha=0.6)
    plt.suptitle(title, fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    if save and path:
        plt.savefig(path)
    plt.show()

