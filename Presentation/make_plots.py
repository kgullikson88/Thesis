from __future__ import print_function, division, absolute_import

from scipy.interpolate import interp1d
from scipy.optimize import minimize_scalar
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
import os
from kglib.spectral_type import SpectralTypeRelations
import pysynphot as S
home = os.environ['HOME']

sns.set_context('talk', font_scale=1.5)
sns.set_style('white')


def powerlaw(q, gamma):
    return (1-gamma) * q**(-gamma)


def expected_mrd_close_vs_wide():
    plt.xkcd()
    q = np.linspace(0, 1, 100)
    wide = powerlaw(q, gamma=0.3)
    close = powerlaw(q, gamma=0.1)

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.plot(q, wide, label='Wide Companions')
    ax.plot(q, close, label='Close Companions')
    ax.set_xlabel('Mass Ratio $M_1/M_2$')
    ax.set_ylabel('P(q)')
    leg = ax.legend(loc='best', fancybox=True)

    plt.show()

#TODO: Plot as a log-normal!
def Gstar_lognormal_plot(mu=5.03, sigma=2.28, mass=1.0):
    P = np.linspace(1e-3, 1e4, 1000)
    logP = np.log10(P*365.25)
    #logP = np.linspace(-2, 10, 100)
    #P = 10**(logP - np.log10(365.25))
    print(P)
    prob = 1/np.sqrt(2*np.pi*sigma**2) * np.exp(-0.5*(logP-mu)**2 / sigma**2)
    
    # Convert to AU
    loga = 2/3 * logP - 2/3*np.log10(365.25) + 1/3 * np.log10(mass)

    # Make the base plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    ax.plot(P, prob)
    ax.set_xlabel('Period (years)')
    ax.set_ylabel('PDF')

    # Add the semimajor axis on top
    top = ax.twiny()
    
    # Set the ticks at the values corresponding to the right period
    #loga_ticks = np.linspace(-2, 5, 8)
    #logP_ticks = 1.5 * loga_ticks - 0.5*np.log10(mass) + np.log10(365.25)
    a_ticks = np.linspace(0, 1000, 11)
    P_ticks = np.sqrt(a_ticks**2 / mass)
    #print(loga_ticks)
    #print(logP_ticks)
    print(a_ticks)
    print(P_ticks)

    top.set_xticks(P_ticks)
    top.set_xticklabels(['{}'.format(a) for a in a_ticks])
    top.set_xlabel('Semimajor axis (AU)')

    # Set the full range to be the same as the data axis
    xlim = ax.get_xlim()
    top.set_xlim(xlim)

    fig.savefig('Figures/Separation_Gstar.pdf')
    plt.show()


def plot_smoothing_method(vsini=25):
    from kglib.utils import HelperFunctions
    
    filename = os.path.join(home, 'School', 'Research', 'CHIRON_data', '20150211', 'HIP_21589.fits')
    orders = HelperFunctions.ReadExtensionFits(filename)

    # Grab one order
    i = HelperFunctions.FindOrderNums(orders, [645])[0]
    order = orders[i]

    # Smooth
    smoothed = HelperFunctions.astropy_smooth(order, vel=vsini, linearize=True)

    # Plot
    fig, (top, bottom) = plt.subplots(2, 1, gridspec_kw={'height_ratios':[3,1]}, sharex=True)
    top.plot(order.x, order.y, 'k-', alpha=0.5)
    top.plot(order.x, smoothed, 'r-', alpha=0.7)
    bottom.plot(order.x, order.y - smoothed, 'k-', alpha=0.5)

    # Label
    bottom.set_xlabel('Wavelength (nm)')
    bottom.set_ylabel('Residuals')
    top.set_ylabel('Flux (arbitrary units)')

    plt.show()

def plot_ccf_search_grid(highT_extension=True, **kwargs):
    fig, ax = plt.subplots(1, 1)
    for T in range(3000, 12000, 100):
        for vsini in [0, 10, 20, 30]:
            ax.plot(T, vsini, 'bo', **kwargs)

    if highT_extension:
        for T in range(9000, 30000, 1000):
            for vsini in [0, 10, 20, 30, 40, 50]:
                if T < 12000 and vsini < 40:
                    continue

                ax.plot(T, vsini, 'bo', **kwargs)
    ax.set_xlabel('Temperature (K)')
    ax.set_ylabel('vsini (km/s)')

    plt.show()


DEFAULT_FILE = os.path.join(home, 'School', 'Research', 'HET_data', 
                           'Cross_correlations', 'CCF_primary_nobalmer.hdf5')
DEFAULT_PATH = 'HIP 97870/2013-08-13/T17000_logg4.5_metal0.0_addmode-simple_vsini150'
DEFAULT_STARNAME = 'HIP 97870'
def plot_primary_ccf(**kwargs):
    # Get the data
    import h5py
    hdf5_filename = kwargs.get('filename', DEFAULT_FILE)
    path = kwargs.get('h5path', DEFAULT_PATH)
    starname = kwargs.get('star', DEFAULT_STARNAME)

    with h5py.File(hdf5_filename, 'r') as f:
        ds = f[path]
        vel, corr = ds.value
        attrs = dict(ds.attrs)

    # Plot
    fig, ax = plt.subplots(1, 1)
    ax.plot(vel, corr, **kwargs)
    ax.set_xlabel('Velocity (km/s)')
    ax.set_ylabel('CCF Power')

    # Annotate
    text = """
           Model Parameters:
           -----------------
           Teff: {} K
           [Fe/H]: {:.1f}
           log(g): 4.0
           vsini: {} km/s
           """.format(attrs['T'], attrs['[Fe/H]'], attrs['vsini'])
    ax.text(x=0.7, y=0.5, s=text, fontdict=dict(size=15))
    ax.set_xlim((-1000, 1000))
    ax.set_title(starname)
    plt.show()


def plot_isochrones(age_values=None, **kwargs):
    from isochrones.padova import Padova_Isochrone
    pad = Padova_Isochrone()

    if age_values is None:
        age_values = [10, 30, 100, 300, 1000]

    fig, ax = plt.subplots(1, 1)
    for age in age_values:
        iso = pad.isochrone(age=np.log10(age)+6)
        ax.plot(iso.Teff, iso.logL, label='{} Myr'.format(age))

    ax.set_xlim(ax.get_xlim()[::-1])
    ax.set_xlabel('Temperature (K)')
    ax.set_ylabel('log(L/Lsun)')
    ax.legend(loc='best', fancybox=True)

    return fig, ax


def plot_isochrones2(age_values=None, mass_values=None, **kwargs):
    from isochrones import padova
    df = padova.MASTERDF

    if age_values is None:
        age_values = [7.0, 7.3, 7.6, 8.0]
    if mass_values is None:
        mass_values = [3.0, 4.0, 5.0, 6.0]


    fig, ax = plt.subplots(1, 1)
    for age in age_values:
        iso = df.loc[((df.age - age)**2 < 1e-5) & (df.feh**2 < 1e-5)]
        ax.plot(10**iso.logTeff, iso.logg, '-', label='{:.0f} Myr'.format(10**(age-6)))

    for mass in mass_values:
        ev = df.loc[((df.M_ini - mass)**2 < 1e-5) & (df.feh**2 < 1e-5)]
        ax.plot(10**ev.logTeff, ev.logg, '--', label='{:.1f} Msun'.format(mass))

    ax.set_xlim(ax.get_xlim()[::-1])
    ax.set_xlabel('Temperature (K)')
    ax.set_ylabel('log(g) (cgs)')
    ax.legend(loc='best', fancybox=True)

    return fig, ax



def plot_q_samples():
    import h5py
    filename = os.path.join(home, 'School', 'Research', 'BinaryInference', 
                            'MCMC_Samples', 'mcmc_samples', 'MassSamples.h5')
    systems = [u'HIP 100221 - AB', u'HIP 100907 - AB', u'HIP 103298 - AB', 
               u'HIP 10732 - AB', u'HIP 109139 - AB']
    with h5py.File(filename, 'r') as infile:
        for system in systems:
            star = system.split('-')[0].strip()
            q = infile[system]['companion_isochrone'].value / infile[system]['primary'].value
            
            fig, ax = plt.subplots(1, 1)
            ax.hist(q[~np.isnan(q)], bins=30, normed=True)
            ax.set_xlabel('Mass ratio (q)')
            ax.set_ylabel('P(q)')
            ax.set_title(star)
            sns.despine()

            fig.savefig('Figures/{}_q.png'.format(star.replace(' ', '_')))
            plt.close('all')


def plot_sample_histogram(**kwargs):
    values = np.random.uniform(0, 1, 100)
    bins = 10
    fig, ax = plt.subplots(1, 1, figsize=(3, 2))
    ax.hist(values, bins=bins, **kwargs)
    sns.despine()
    ax.set_xticks([])
    ax.set_yticks([])
    fig.subplots_adjust(bottom=0.15)

    plt.savefig('Figures/SampleHist.png')
    plt.show()




def plot_sample_lognormal(mu=np.log(0.3), sigma=np.log(0.1), **kwargs):
    x = np.arange(0, 1, 0.01)
    y = 1 / (x*np.sqrt(2*np.pi*sigma**2)) * np.exp(-0.5*(np.log(x) - mu)**2 / sigma**2)

    fig, ax = plt.subplots(1, 1, figsize=(3, 2))
    ax.plot(x, y)
    sns.despine()
    ax.set_xticks([])
    ax.set_yticks([])
    #fig.subplots_adjust(bottom=0.15)

    plt.savefig('Figures/SampleLognormal.png')
    plt.show()




def plot_sample_powerlaw(gamma=0.4, **kwargs):
    x = np.arange(0, 1, 0.01)
    y = (1-gamma) * x**(-gamma)

    fig, ax = plt.subplots(1, 1, figsize=(3, 2))
    ax.plot(x, y)
    sns.despine()
    ax.set_xticks([])
    ax.set_yticks([])
    #fig.subplots_adjust(bottom=0.15)

    plt.savefig('Figures/SamplePowerlaw.png')
    plt.show()




def get_magdiff(spt1, spt2):
    MS = SpectralTypeRelations.MainSequence()
    Kmag1 = MS.GetAbsoluteMagnitude(spt1, color='K')
    Kmag2 = MS.GetAbsoluteMagnitude(spt2, color='K')
    return Kmag2 - Kmag1


def convert_magdiff(dM, color1='V', color2='K', p_spt='A0'):
    """
    Converts a magnitude difference in one system to another. Assumes a primary of spectral type p_spt
    """
    # First, get the spectral type of the secondary
    MS = SpectralTypeRelations.MainSequence()
    prim_mag = MS.GetAbsoluteMagnitude(p_spt, color=color1)
    diff = np.inf
    s_spt = 'B0'
    for spec_type in ['B', 'A', 'F', 'G', 'K', 'M']:
        for subtype in np.arange(0, 10.0, 0.1):
            spt = '{}{}'.format(spec_type, subtype)
            mag = MS.GetAbsoluteMagnitude(spt, color=color1)
            if abs(mag - prim_mag - dM) < diff:
                diff = abs(mag - prim_mag - dM)
                s_spt = spt

    # Now, just get the magnitude in the new band
    prim_mag = MS.GetAbsoluteMagnitude(p_spt, color=color2)
    sec_mag = MS.GetAbsoluteMagnitude(s_spt, color=color2)

    print (dM, '-->', sec_mag - prim_mag)
    print ('Secondary spt = {}\n'.format(s_spt))

    return sec_mag - prim_mag


def K1(m2, m1, a, sini=1, e=0):
  return 2.98e4* m2 / m1 * numpy.sqrt(m1 + m2) / numpy.sqrt(a) * sini/numpy.sqrt(1-e**2)


def errfcn(m1, m2, a, precision):
    return abs(precision - K1(m1, m2, a, sini=numpy.sqrt(2)/2.0))

# Interferometry limits from Aldoretta et al 2015AJ....149...26A
int_sep = [0.02, 0.03, 0.06, 0.08, 0.13, 0.2, 0.4]
int_contrast = [1.7, 2.8, 2.9, 4.0, 4.2, 4.5, 4.8]

def make_contrast_curve_plot():
    # Get the average (actually median) VAST limits
    directory = '{}/Dropbox/School/Research/VAST_Survey/DeRosa2014/VAST_Limits'.format(os.environ['HOME'])
    all_limit_files = ['{}/{}'.format(directory, f) for f in os.listdir(directory) if f.endswith('-K')]
    x = np.arange(0.0, 1.0, 0.01)
    y = []
    for fname in all_limit_files:
        sep, contrast = np.loadtxt(fname, unpack=True)
        fcn = interp1d(sep, contrast, bounds_error=False)
        y.append(fcn(x))
    y = np.array(y)
    y_mean = np.nanmedian(y, axis=0)

    #Make rv curve. Kind of weird in this space but meh...
    a = np.logspace(-3,2,100)
    MS = SpectralTypeRelations.MainSequence()
    Kmag_primary = MS.GetAbsoluteMagnitude('A0', color='K')
    filt = S.ObsBandpass('K')
    pri_radius = MS.Interpolate(MS.Radius, pri_spt)
    pri_spec = S.BlackBody(Tprim)
    pri_obs = S.Observation(pri_spec, filt)
    pri_flux = pri_obs.trapezoidIntegration(pri_obs.wave, pri_obs.flux)
    rv_mag_now = []
    for ai in a:
        
        pars=[5.0, ai, precision_now]
        m2 = minimize_scalar(errfcn, args=tuple(pars), bracket=(1e-8,5.0), bounds=(0,5.0), method='Brent').x
        spt = MS.GetSpectralType(MS.Mass, m2, interpolate=True)
        Tsec = MS.Interpolate(MS.Temperature, spt)
        sec_radius = MS.Interpolate(MS.Radius, spt)
        sec_spec = S.BlackBody(Tsec)
        sec_obs = S.Observation(sec_spec, filt)
        sec_flux = sec_obs.trapezoidIntegration(sec_obs.wave, sec_obs.flux)
        delta_mag = -2.5*numpy.log10( (sec_flux * sec_radius**2) / (pri_flux * pri_radius**2) )
        rv_mag_now.append(delta_mag)

    a = a/100.0

    # Make figure
    fig = plt.figure(figsize=(10, 10))
    fs = 20
    ax = fig.add_subplot(111)

    # Plot VAST and interferometry limits
    ax.plot(x, y_mean, 'k-', lw=2, label='Average VAST Survey Limits')
    int_contrast2 = [convert_magdiff(c, 'V', 'K') for c in int_contrast]
    ax.plot(int_sep, int_contrast2, 'k-', lw=2, label='Typical Interferometry')

    # Plot some stars
    G5 = get_magdiff('A0', 'G5')
    K0 = get_magdiff('A0', 'K0')
    K5 = get_magdiff('A0', 'K5')
    M0 = get_magdiff('A0', 'M0')
    M5 = get_magdiff('A0', 'M5')
    ax.plot(x, M0 * np.ones(x.size), 'r--', lw=2, label='A0/M0 binary')
    ax.plot(x, M5 * np.ones(x.size), 'g:', lw=2, label='A0/M5 binary')
    ax.plot(x, G5 * np.ones(x.size), 'b-.', lw=2, label='A0/G5 binary')

    # Reverse y axis
    ax.set_ylim(ax.get_ylim()[::-1])

    # Set labels
    ax.set_xlabel('Separation (")', fontsize=fs)
    ax.set_ylabel('$\Delta K$', fontsize=fs)

    # Legend/labels
    # leg = ax.legend(loc='best', fancybox=True)
    #leg.get_frame().set_alpha(0.5)
    ax.text(0.7, M5 - 0.1, 'A0/M5 binary', fontsize=fs)
    ax.text(0.7, M0 - 0.1, 'A0/M0 binary', fontsize=fs)
    ax.text(0.7, G5 - 0.1, 'A0/G5 binary', fontsize=fs)
    ax.text(0.44, 6.07, 'Average VAST Survey Limits', fontsize=fs, rotation=-28)
    ax.text(0.09, 2.5, 'HST FGS Interferometry', fontsize=fs, rotation=-10)
    # Show the plot
    plt.show()



if __name__ == '__main__':
    expected_mrd_close_vs_wide()