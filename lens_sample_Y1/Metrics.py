import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm
from matplotlib.ticker import FormatStrFormatter
from qp.ensemble import Ensemble
from matplotlib import gridspec
from qp.metrics.pit import PIT
from qp import interp

def plot_scatter(zphot,
                 ztrue,
                 zmin=0,
                 zmax=3,
                 bins=150,
                 cmap='viridis',
                 line_color='red',
                 line_width=0.2,
                 title='$z_{true}$ vs $z_{phot}$',
                 xlabel='z$_{true}$',
                 ylabel='z$_{phot}$', 
                 fontsize_title=18,
                 fontsize_labels=15,
                 path_to_save=''):
    """
    Plot a histogram of photometric redshift vs true redshift with a diagonal line.

    Parameters
    ----------
    zphot : array-like
        Array of photometric redshifts.
    ztrue : array-like
        Array of true (spectroscopic) redshifts.
    zmin : float, optional
        Minimum redshift value for the plot axes.
    zmax : float, optional
        Maximum redshift value for the plot axes.
    bins : int, optional
        Number of bins for the histogram.
    cmap : str, optional
        Colormap to be used for the histogram.
    line_color : str, optional
        Color of the diagonal line.
    line_width : float, optional
        Width of the diagonal line.
    title : str, optional
        Title of the plot.
    xlabel : str, optional
        Label for the x-axis.
    ylabel : str, optional
        Label for the y-axis.
    fontsize_title : int, optional
        Font size for the title.
    fontsize_labels : int, optional
        Font size for the x and y labels.
    path_to_save : str, optional
        Path to save the plot image.

    Returns
    -------
    None
    """
    sns.histplot(x=ztrue, y=zphot, bins=bins, cmap=cmap)
    plt.plot([0,3], [0,3], color=line_color, linewidth=line_width)
    plt.xlim(zmin, zmax)
    plt.ylim(zmin, zmax)
    plt.xlabel(xlabel, fontsize=fontsize_labels)
    plt.ylabel(ylabel, fontsize=fontsize_labels)
    plt.title(title, fontsize=fontsize_title)
    
    if path_to_save:
        plt.savefig(path_to_save)
    
    plt.show()

    
    


def plot_old_valid(zphot, 
                   ztrue,
                   title='', 
                   bins=150,
                   gals=None,
                   z_max=3.0,
                   cmap='inferno',
                   colors=None,
                   code="",
                   save=False,
                   path_to_save=''):
    """
    Plot traditional Zphot vs. Zspec and N(z) plots for illustration.
    Ancillary function to be used by class Sample.

    Parameters
    ----------
    zphot : array-like
        Array of photometric redshifts.
    ztrue : array-like
        Array of true (spectroscopic) redshifts.
    title : str, optional
        Title of the plot.
    gals : list, optional
        List of galaxies' indexes to highlight.
    colors : list, optional
        List of HTML codes for colors used in the plot for highlighted points.
    code : str, optional
        Code or label to be used in the legend.
    path_to_save : str, optional
        Path to save the plot image.

    Returns
    -------
    None
    """
    df = pd.DataFrame({'z$_{true}$': ztrue, 'z$_{phot}$': zphot})
    fig = plt.figure(figsize=(10, 4), dpi=100)
    
    ax = plt.subplot(121)
    
    counts, xedges, yedges = np.histogram2d(zphot,ztrue, bins=120)
    norm = LogNorm(vmin=counts[counts > 0].min(), vmax=counts.max())
    sns.heatmap(
        counts.T, 
        cmap='inferno', 
        norm=norm, 
        xticklabels=False, 
        cbar=False,
        yticklabels=False
    )

    plt.axhline(y=0, color='black', linewidth=1.5)  
    plt.axvline(x=0, color='black', linewidth=2.5)  
    plt.axvline(x=counts.shape[0], color='black', linewidth=2.5)
    plt.axhline(y=counts.shape[1], color='black', linewidth=1.5)
    
    plt.xlim(0, counts.shape[0])
    plt.ylim(0, counts.shape[1])

    plt.xticks(ticks=np.linspace(0, counts.shape[0], 4), labels=np.linspace(0, 3, 4), fontsize=14)
    plt.yticks(ticks=np.linspace(0, counts.shape[1], 4), labels=np.linspace(0, 3, 4), fontsize=14)
    plt.xlabel('z$_{phot}$', fontsize=20)
    plt.ylabel('z$_{true}$', fontsize=20)

    plt.tight_layout()
    

    plt.subplot(122)
    sns.kdeplot(ztrue, shade=False, label='z$_{true}$', bw_adjust=0.8, color='dodgerblue')
    sns.kdeplot(zphot, shade=False, label='z$_{phot}$', bw_adjust=0.8, color='orangered')
    plt.xlim(0, z_max)
    plt.xlabel('z', fontsize=22)
    plt.ylabel('Density', fontsize=20)
    plt.legend(fontsize=20)
    plt.tick_params(axis='both', which='major', labelsize=18)
    plt.suptitle(title, fontsize=20)
    plt.tight_layout()
    
    if save:
        plt.savefig(path_to_save)
        
        
         
def plot_metrics(zspec,
                 zphot,
                 maximum=3,
                 save=False,
                 path_to_save='',
                 title=None,
                 xlim=3,
                 ylim=[-0.025,0.025],
                 initial=0):
    """
    Plot Bias, Sigma_68, Out2σ, Out3σ given spectroscopic and photometric redshifts.

    Parameters
    ----------
    zspec : array-like
        Array of spectroscopic redshifts.
    zphot : array-like
        Array of photometric redshifts.
    maximum : float, optional
        Maximum redshift value for the plots.
    path_to_save : str, optional
        Path to save the plot image.
    title : str, optional
        Title of the plot.
    initial : float, optional
        Minimum redshift value for the plots.

    Returns
    -------
    None
    """
    bins = np.arange(initial, maximum, 0.1)
    points = bins + 0.05
    fraction_outliers = []
    sigma68z = []
    sigmaz = []
    meanz = []
    outliers_2 = []
    fr_e = []

    for index in range(len(bins) - 1):
        bin_lower = bins[index]
        bin_upper = bins[index + 1]
        
        values_r = zphot[(zphot >= bin_lower) & (zphot <= bin_upper)]
        values_s = zspec[(zphot >= bin_lower) & (zphot <= bin_upper)]
        
        deltabias = (values_r - values_s) / (1 + values_s)
        mean_bias = np.mean(deltabias)
        meanz.append(mean_bias)
        
        s = np.sort(np.abs(deltabias / (1 + values_s)))
        if len(s)==0:
            sigma68z.append(0)
            fr_e.append(0)
            sigmaz.append(0)
            fraction_outliers.append(0)
            outliers_2.append(0)
            continue
        
        sigma68 = s[int(len(s) * 0.68)]
        sigma68z.append(sigma68)    
        
        sigma = (np.sum((values_r - values_s - mean_bias) ** 2) / len(values_r)) ** 0.5
        sigmaz.append(sigma)
        
        outliers = deltabias[np.abs(deltabias - mean_bias) > 3 * sigma]
        fraction_outlier = len(outliers) / len(deltabias)
        fraction_outliers.append(fraction_outlier)
        
        outliers2 = deltabias[np.abs(deltabias - mean_bias) > 2 * sigma]
        fraction_outlier2 = len(outliers2) / len(deltabias)
        outliers_2.append(fraction_outlier2)

    fig, axes = plt.subplots(4, 1, figsize=(15, 8), sharex=True)
    plt.subplots_adjust(hspace=0.001)
    
    x_lim = (0, np.max(bins))

    # Mean Bias plot
    axes[0].plot(points[:-1], meanz, 'bo-')
    axes[0].axhline(0, color='black', linestyle='--')
    axes[0].set_ylabel(r'$\Delta z$', fontsize=28)
    axes[0].set_xlim(x_lim)
    axes[0].set_ylim(ylim)
    axes[0].tick_params(axis='both', labelsize=18)
    axes[0].grid(True)
    axes[0].legend(fontsize=16)

    # Sigma 68 plot
    axes[1].plot(points[:-1], sigma68z, 'go-')
    axes[1].set_ylabel(r'$\sigma_{68}$', fontsize=28)
    axes[1].set_xlim(x_lim)
    axes[1].set_ylim(0, 0.03)
    axes[1].tick_params(axis='both', labelsize=18)
    axes[1].grid(True)

    # Fraction of outliers beyond 2σ plot
    axes[2].plot(points[:-1], outliers_2, 'o-', color='darkorange')
    axes[2].set_ylabel('out$_{2σ}$', fontsize=28)
    axes[2].set_xlim(x_lim)
    axes[2].set_ylim(0, 0.08)
    axes[2].tick_params(axis='both', labelsize=18)
    axes[2].grid(True)
    axes[2].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    # Fraction of outliers beyond 3σ plot
    axes[3].plot(points[:-1], fraction_outliers, 'ro-')
    axes[3].set_xlabel(r'z$_{phot}$', fontsize=28)
    axes[3].set_ylabel('out$_{3σ}$', fontsize=28)
    axes[3].set_xlim(x_lim)
    axes[3].set_ylim(0, 0.015)
    axes[3].tick_params(axis='both', labelsize=18)
    axes[3].grid(True)

    plt.suptitle(title, fontsize=32)
    plt.xlim(0, xlim)
    plt.tight_layout()

    if save:
        plt.savefig(f'{path_to_save}')

    plt.show()





    
def plot_moments_n(hist_true, hist_n, title, path_to_save=''):


    nz = hist_n[0]
    zgrid_n = hist_n[1]
    
    z = hist_true[0]
    zgrid_z = hist_true[1]
    

    mean_n = np.average(zgrid_n, weights=nz)
    variance_n = np.average((zgrid_n - mean_n) ** 2, weights=nz)
    skew_n = np.average(((zgrid_n - mean_n) / np.sqrt(variance_n))**3, weights=nz)

    mean_z = np.average(zgrid_z, weights=z)
    variance_z = np.average((zgrid_z - mean_z) ** 2, weights=z)
    skew_z = np.average(((zgrid_z - mean_z) / np.sqrt(variance_z))**3, weights=z)
    
    fig, ax1 = plt.subplots(figsize=(12, 8))
    
    ax1.axhline(0,ls='--',color='black')
    ax1.axhline(-0.005,ls='--',color='blue')
    ax1.axhline(0.005,ls='--',color='blue', label = 'Y10 SRD Req variance' )
    ax1.axhline(-0.003,ls='--',color='red', label = 'Y10 SRD Req mean' )
    ax1.axhline(0.003,ls='--',color='red')

    ax1.set_ylabel('$Δ⟨z^m⟩$', fontsize=20)
    ax1.scatter(1,mean_n-mean_z, color='red', marker='D', s=70,label='Mean (m=1)')
    ax1.scatter(1,variance_n-variance_z, color='blue', marker='*', s=70,label='Variance (m=2)')
    ax1.scatter(1,skew_n-skew_z, color='black', marker='X', s=70, label='Skewness (m=3)')
      
    
    ax1.set_ylim(-0.05,0.05)
    ax1.tick_params(labelbottom = False)
    ax1.set_xlim(0.75, 1.25)
    ax1.legend(fontsize = 15)

    plt.suptitle(f'{title}', fontsize=16)
    plt.tight_layout()
    
    if path_to_save=='':
        None
    else:
        plt.savefig(path_to_save)
        
    plt.show()
    return [mean_z-mean_n,variance_z-variance_n,skew_z-skew_n]   

def ks_plot(pitobj, n_quant=100):
    """ KS test illustration.
    Ancillary function to be used by class KS."""
    pits = np.array(pitobj.pit_samps)
    stat_and_pval = pitobj.evaluate_PIT_KS()
    xvals = np.linspace(0., 1., n_quant)
    yvals = np.array([np.histogram(pits, bins=len(xvals))[0]])
    pit_cdf = Ensemble(interp, data=dict(xvals=xvals, yvals=yvals)).cdf(xvals)[0]
    uniform_yvals = np.array([np.full(n_quant, 1.0 / float(n_quant))])
    uniform_cdf = Ensemble(interp, data=dict(xvals=xvals, yvals=uniform_yvals)).cdf(xvals)[0]

    plt.figure(figsize=[4, 4])
    plt.plot(xvals, uniform_cdf, '--', label="CDF z$_{true}$", color = 'dodgerblue')
    plt.plot(xvals, pit_cdf, 'b-', label="CDF z$_{spec}$", color='orangered')
    bin_stat = np.argmax(np.abs(pit_cdf - uniform_cdf))

    plt.vlines(x=xvals[bin_stat],
               ymin=np.min([pit_cdf[bin_stat], uniform_cdf[bin_stat]]),
               ymax=np.max([pit_cdf[bin_stat], uniform_cdf[bin_stat]]),
               colors='k')
    plt.plot(xvals[bin_stat], pit_cdf[bin_stat], "k.")
    plt.plot(xvals[bin_stat], uniform_cdf[bin_stat], "k.")
    ymean = (pit_cdf[bin_stat] + uniform_cdf[bin_stat]) / 2.
    plt.text(xvals[bin_stat] + 0.05, ymean, "max", fontsize=16)
    plt.xlabel("PIT value")
    plt.ylabel("CDF(PIT)")
    xtext = 0.57
    ytext = 0.05
    bbox = dict(boxstyle='round', fc='blanchedalmond', ec='orange', alpha=0.3)
    plt.text(xtext, ytext, f"KS = {stat_and_pval.statistic:.4f}", fontsize=14, bbox=bbox)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.legend(fontsize = 12)
    plt.tight_layout()
    
def plot_pit_qq(pdfs, zgrid, ztrue, bins=None, title=None, code=None,
                show_pit=True, show_qq=True,
                pit_out_rate=None, savefig=False, path_to_save='') -> str:
    """Quantile-quantile plot
    Ancillary function to be used by class Metrics.

    Parameters
    ----------
    pit: `PIT` object
        class from metrics.py
    bins: `int`, optional
        number of PIT bins
        if None, use the same number of quantiles (sample.n_quant)
    title: `str`, optional
        if None, use formatted sample's name (sample.name)
    label: `str`, optional
        if None, use formatted code's name (sample.code)
    show_pit: `bool`, optional
        include PIT histogram (default=True)
    show_qq: `bool`, optional
        include QQ plot (default=True)
    pit_out_rate: `ndarray`, optional
        print metric value on the plot panel (default=None)
    savefig: `bool`, optional
        save plot in .png file (default=False)
    """

    if bins is None:
        bins = 100
    if title is None:
        title = ""

    if code is None:
        code = ""
        label = ""
    else:
        label = code + "\n"


    if pit_out_rate is not None:
        try:
            label += "PIT$_{out}$: "
            label += f"{float(pit_out_rate):.4f}"
        except:
            print("Unsupported format for pit_out_rate.")

    plt.figure(figsize=[4, 5])
    gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1])
    ax0 = plt.subplot(gs[0])
    sample = Sample(pdfs, zgrid, ztrue)

    if show_qq:
        ax0.plot(sample.qq[0], sample.qq[1], c='r',
                 linestyle='-', linewidth=3, label=label)
        ax0.plot([0, 1], [0, 1], color='k', linestyle='--', linewidth=2)
        ax0.set_ylabel("Q$_{data}$", fontsize=18)
        plt.ylim(-0.001, 1.001)
    plt.xlim(-0.001, 1.001)
    plt.title(title,fontsize=20)
    if show_pit:
        fzdata = Ensemble(interp, data=dict(xvals=zgrid, yvals=pdfs))
        pitobj = PIT(fzdata, ztrue)
        pit_vals = np.array(pitobj.pit_samps)
        pit_out_rate = pitobj.evaluate_PIT_outlier_rate()

        try:
            y_uni = float(len(pit_vals)) / float(bins)
        except:
            y_uni = float(len(pit_vals)) / float(len(bins))
        if not show_qq:
            ax0.hist(pit_vals, bins=bins, alpha=0.7, label=label)
            ax0.set_ylabel('Number')
            ax0.hlines(y_uni, xmin=0, xmax=1, color='k')
            plt.ylim(0, )  # -0.001, 1.001)
        else:
            ax1 = ax0.twinx()
            ax1.hist(pit_vals, bins=bins, alpha=0.7)
            ax1.set_ylabel('Number')
            ax1.hlines(y_uni, xmin=0, xmax=1, color='k')
    leg = ax0.legend(handlelength=0, handletextpad=0, fancybox=True)
    if show_qq:
        ax2 = plt.subplot(gs[1])
        ax2.plot(sample.qq[0], (sample.qq[1] - sample.qq[0]), c='r', linestyle='-', linewidth=3)
        plt.ylabel(r"$\Delta$Q", fontsize=18)
        ax2.plot([0, 1], [0, 0], color='k', linestyle='--', linewidth=2)
        plt.xlim(-0.001, 1.001)
        plt.ylim(np.min([-0.12, np.min(sample.qq[1] - sample.qq[0]) * 1.05]),
                 np.max([0.12, np.max(sample.qq[1] - sample.qq[0]) * 1.05]))
    if show_pit:
        if show_qq:
            plt.xlabel("Q$_{theory}$ / PIT Value", fontsize=18)
        else:
            plt.xlabel("PIT Value", fontsize=18)
    else:
        if show_qq:
            plt.xlabel("Q$_{theory}$", fontsize=18)
    if savefig:
        plt.tight_layout()
        plt.savefig(path_to_save)


class Sample(Ensemble):
    """ Expand qp.Ensemble to append true redshifts
    array, metadata, and specific plots. """

    def __init__(self, pdfs, zgrid, ztrue, photoz_mode=None, code="", name="", n_quant=100):
        """Class constructor

        Parameters
        ----------
        pdfs: `ndarray`
            photo-z PDFs array, shape=(Ngals, Nbins)
        zgrid: `ndarray`
            PDF bins centers, shape=(Nbins,)
        ztrue: `ndarray`
            true redshifts, shape=(Ngals,)
        photoz_mode: `ndarray`
            photo-z (PDF mode), shape=(Ngals,)
        code: `str`, (optional)
            algorithm name (for plot legends)
        name: `str`, (optional)
            sample name (for plot legends)
        """

        super().__init__(interp, data=dict(xvals=zgrid, yvals=pdfs))
        self._pdfs = pdfs
        self._zgrid = zgrid
        self._ztrue = ztrue
        self._photoz_mode = photoz_mode
        self._code = code
        self._name = name
        self._n_quant = n_quant
        self._pit = None
        self._qq = None


    @property
    def code(self):
        """Photo-z code/algorithm name"""
        return self._code

    @property
    def name(self):
        """Sample name"""
        return self._name

    @property
    def ztrue(self):
        """True redshifts array"""
        return self._ztrue

    @property
    def zgrid(self):
        """Redshift grid (binning)"""
        return self._zgrid

    @property
    def photoz_mode(self):
        """Photo-z (mode) array"""
        return self._photoz_mode

    @property
    def n_quant(self):
        return self._n_quant

    @property
    def pit(self):
        if self._pit is None:
            pit_array = np.array([self[i].cdf(self.ztrue[i])[0][0] for i in range(len(self))]) 
            self._pit = pit_array
        return self._pit

    @property
    def qq(self, n_quant=100):
        q_theory = np.linspace(0., 1., n_quant)
        q_data = np.quantile(self.pit, q_theory)
        self._qq = (q_theory, q_data)
        return self._qq

    def __len__(self):
        if len(self._ztrue) != len(self._pdfs):
            raise ValueError("Number of pdfs and true redshifts do not match!!!")
        return len(self._ztrue)

    def __str__(self):
        code_str = f'Algorithm: {self._code}'
        name_str = f'Sample: {self._name}'
        line_str = '-' * (max(len(code_str), len(name_str)))
        text = str(line_str + '\n' +
                   name_str + '\n' +
                   code_str + '\n' +
                   line_str + '\n' +
                   f'{len(self)} PDFs with {len(self.zgrid)} probabilities each \n' +
                   f'qp representation: {self.gen_class.name} \n' +
                   f'z grid: {len(self.zgrid)} z values from {np.min(self.zgrid)} to {np.max(self.zgrid)} inclusive')
        return text