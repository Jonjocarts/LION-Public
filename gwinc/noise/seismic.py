'''Functions to calculate seismic noise in suspended optics.

'''
from __future__ import division
import numpy as np
from scipy.interpolate import PchipInterpolator as interp1d


def seismic_suspension_fitered(sus, in_trans):
    """Seismic displacement noise for single suspended test mass.

    :sus: gwinc suspension structure
    :in_trans: input translational displacement spectrum

    :returns: tuple of displacement noise power spectrum at :f:, and
    horizontal and vertical components.

    """
    hTable = sus.hTable
    vTable = sus.vTable

    theta = sus.VHCoupling.theta

    # horizontal noise total
    nh = (abs(hTable)**2) * in_trans**2

    # vertical noise total
    nv = (abs(theta * vTable)**2) * in_trans**2

    # new total noise
    n = nv + nh

    return n, nh, nv


def seismic_BSC_ISI(f):
    """Rough seismic noise spectra on aLIGO BSC ISI table.

    :f: frequency array in Hz

    :returns: tuple of displacement noise power spectrum at :f: for
    translational and rotational DOFs.

    """
    SEI_F = np.array([0.01, 0.03, 0.1, 0.2, 0.5, 1, 10, 30, 300])

    # translational DOFs
    SEI_T = np.array([3e-6, 1e-6, 2e-7, 2e-7, 8e-10, 1e-11, 3e-13, 3e-14, 3e-14])
    nt = 10**(interp1d(SEI_F, np.log10(SEI_T))(f))

    # rotational DOFs
    SEI_R = np.array([1e-8, 3e-8, 2e-8, 1e-8, 4e-10, 1e-11, 3e-13, 3e-14, 3e-14])
    nr = 10**(interp1d(SEI_F, np.log10(SEI_R))(f))

    return nt, nr


def seismic_BSC_ISI_6D(f):
    """Rough seismic noise spectra on aLIGO BSC ISI table with a 6D seismometer.

    This largely follows Mow-Lowry and Martynov, arXiv:1801.01468.

    :f: frequency array in Hz

    :returns: tuple of displacement noise power spectrum at :f: for
    translational and rotational DOFs.

    """
    # FIXME: merge this with above, using flag

    SEI_F = np.array([0.01, 0.03, 0.1, 0.2, 0.5, 1, 10, 100, 300])

    SEI_T_self = np.array([1e-7, 1e-9, 3e-11, 6e-12, 3e-13, 1e-13, 3e-14, 1e-14, 1e-14])/1000
    nt_self = 10**(interp1d(SEI_F, np.log10(SEI_T_self))(f))
    nt_gnd = 10*seismic_ground_NLNM(f)
    blend_t = np.abs(100/(1+1j*f/0.01)**4)
    nt = np.sqrt(nt_self**2 + (blend_t * nt_gnd)**2)

    SEI_R_self = np.array([2e-11, 5e-12, 1e-12, 6e-13, 3e-13, 2e-13, 6e-14, 2e-14, 2e-14])/1000
    nr_self = 10**(interp1d(SEI_F, np.log10(SEI_R_self))(f))
    nr_gnd = np.abs(1e-7/(1+1j*f/0.001))
    blend_r = np.abs(100/(1+1j*f/0.01)**4)
    nr = np.sqrt(nr_self**2 + (blend_r * nr_gnd)**2)

    return nt, nr


def seismic_ground_NLNM(f):
    """The Peterson new generic ground motion low noise model.

    :f: frequency array in Hz

    :returns: displacement noise amplitude spectrum at :f:

    """
    Pl = np.array([
        1.00e-02, 1.00e-01, 1.70e-01, 4.00e-01, 8.00e-01, 1.24e+00,
        2.40e+00, 4.30e+00, 5.00e+00, 6.00e+00, 1.00e+01, 1.20e+01,
        1.56e+01, 2.19e+01, 3.16e+01, 4.50e+01, 7.00e+01, 1.01e+02,
        1.54e+02, 3.28e+02, 6.00e+02, 1.00e+04])
    Al = np.array([
        -156.72, -162.36, -166.7, -170.0, -166.4, -168.6, -159.98,
        -141.1, -71.36, -97.26, -132.18, -205.27, -37.65, -114.37,
        -160.58, -187.5, -216.47, -185.0, -168.34, -217.43, -258.28,
        -346.88])
    Bl = np.array([
        5.64, 5.64, 0.0, -8.3, 28.9, 52.48, 29.81,
        0.0, -99.77, -66.49, -31.57, 36.16, -104.33, -47.1,
        -16.28, 0.0, 15.7, 0.0, -7.61, 11.9, 26.6,
        48.75])
    nlnm = 10**(np.interp(1/f, Pl, Al+Bl*np.log10(Pl))/20) / (2 * np.pi * f)**2
    return nlnm


def seismic_ground_NHNM(f):
    """The Peterson new generic ground motion high noise model.

    :f: frequency array in Hz

    :returns: displacement noise amplitude spectrum at :f:

    """

    Pl = np.array([
        1.00e-01, 2.20e-01, 3.20e-01, 8.00e-01, 3.80e+00,
        4.60e+00, 6.30e+00, 7.90e+00, 1.54e+01, 2.00e+01,
        3.54e+02,
        ])
    Al = np.array([
        -108.73, -150.34, -122.31, -116.85, -108.48,
        -74.66, 0.66, -93.37, 73.54, -151.52,
        -206.66,
        ])
    Bl = np.array([
        -17.23, -80.50, -23.87, 32.51, 18.08,
        -32.95, -127.18, -22.42, -162.98, 10.01,
        31.63,
        ])
    nhnm = 10**(np.interp(1/f, Pl, Al+Bl*np.log10(Pl))/20) / (2 * np.pi * f)**2
    return nhnm
