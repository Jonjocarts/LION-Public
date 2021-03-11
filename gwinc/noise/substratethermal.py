'''Functions to calculate substrate thermal noise

'''
from __future__ import division, print_function
import numpy as np
from numpy import exp, inf, pi, sqrt
import scipy.special
import scipy.integrate

from .. import const
from ..const import BESSEL_ZEROS as zeta
from ..const import J0M as j0m


def substrate_thermorefractive(f, materials, wBeam, exact=False):
    """Substrate thermal displacement noise spectrum from thermorefractive fluctuations

    :f: frequency array in Hz
    :materials: gwinc optic materials structure
    :wBeam: beam radius (at 1 / e^2 power)
    :exact: whether to use adiabatic approximation or exact calculation (False)

    :returns: displacement noise power spectrum at :f:, in meters

    """
    H = materials.MassThickness
    kBT = const.kB * materials.Substrate.Temp
    Temp = materials.Substrate.Temp
    rho = materials.Substrate.MassDensity
    beta = materials.Substrate.dndT
    C = materials.Substrate.MassCM
    kappa = materials.Substrate.MassKappa
    r0 = wBeam/np.sqrt(2)
    omega = 2*pi*f

    if exact:
        def integrand(k, om, D):
            return D * k**3 * exp(-k**2 * wBeam**2/4) / (D**2 * k**4 + om**2)

        inte = np.array([scipy.integrate.quad(lambda k: integrand(k, om, kappa/(rho*C)), 0, inf)[0] for om in omega])

        # From P1400084 Heinert et al. Eq. 15
        #psdCD = @(gamma,m,int) 2*(3/pi^7)^(1/3)*kBT*H*gamma^2*m/hbar^2*cdDens^(1/3)*int; %units are meters
        psdTR = lambda int_: 2/pi * H * beta**2 * kBT * Temp / (rho*C) * int_

        psd = psdTR(inte)
        psd = 2/pi * H * beta**2 * kBT * Temp / (rho*C) * inte

    else:
        psd = 4*H*beta**2*kappa*kBT*Temp/(pi*r0**4*omega**2*(rho*C)**2)

    return psd


def substrate_brownian(f, materials, wBeam):
    """Substrate thermal displacement noise spectrum due to substrate mechanical loss

    :f: frequency array in Hz
    :materials: gwinc optic materials structure
    :wBeam: beam radius (at 1 / e^2 power)

    :returns: displacement noise power spectrum at :f:, in meters

    """
    Y = materials.Substrate.MirrorY
    sigma = materials.Substrate.MirrorSigma
    c2 = materials.Substrate.c2
    n = materials.Substrate.MechanicalLossExponent
    alphas = materials.Substrate.Alphas
    kBT = const.kB * materials.Substrate.Temp

    cftm, aftm = substrate_brownian_FiniteCorr(materials, wBeam)

    # Bulk substrate contribution
    phibulk = c2 * f**n
    cbulk = 8 * kBT * aftm * phibulk / (2 * pi * f)

    # Surface loss contribution
    # csurf = alphas/(Y*pi*wBeam^2)
    csurf = alphas*(1-2*sigma)/((1-sigma)*Y*pi*wBeam**2)
    csurf *= 8 * kBT / (2 * pi * f)

    return csurf + cbulk


def substrate_brownian_FiniteCorr(materials, wBeam):
    """Substrate brownian noise finite-size test mass correction

    :materials: gwinc optic materials structure
    :wBeam: beam radius (at 1 / e^2 power)

    :returns: correction factors tuple:
    cftm = finite mirror correction factor
    aftm = amplitude coefficient for thermal noise:
           thermal noise contribution to displacement noise is
           S_x(f) = (8 * kB * T / (2*pi*f)) * Phi(f) * aftm

    Equation references to Bondu, et al. Physics Letters A 246 (1998)
    227-236 (hereafter BHV) and Liu and Thorne gr-qc/0002055 (hereafter LT)

    """
    a = materials.MassRadius
    h = materials.MassThickness
    Y = materials.Substrate.MirrorY
    sigma = materials.Substrate.MirrorSigma

    # LT uses e-folding of power
    r0 = wBeam / sqrt(2)
    km = zeta/a

    Qm = exp(-2*km*h) # LT eq. 35a

    Um = (1-Qm)*(1+Qm)+4*h*km*Qm
    Um = Um/((1-Qm)**2-4*(km*h)**2*Qm) # LT 53 (BHV eq. btwn 29 & 30)

    x = exp(-(zeta*r0/a)**2/4)
    s = sum(x/(zeta**2*j0m)) # LT 57

    x2 = x*x
    U0 = sum(Um*x2/(zeta*j0m**2))
    U0 = U0*(1-sigma)*(1+sigma)/(pi*a*Y) # LT 56 (BHV eq. 3)

    p0 = 1/(pi*a**2) # LT 28
    DeltaU = (pi*h**2*p0)**2
    DeltaU = DeltaU + 12*pi*h**2*p0*sigma*s
    DeltaU = DeltaU + 72*(1-sigma)*s**2
    DeltaU = DeltaU*a**2/(6*pi*h**3*Y) # LT 54

    # LT 58 (eq. following BHV 31)
    aftm = DeltaU + U0

    # amplitude coef for infinite TM, LT 59
    # factored out: (8 * kB * T * Phi) / (2 * pi * f)
    aitm = (1 - sigma**2) / (2 * sqrt(2 * pi) * Y * r0)

    # finite mirror correction
    cftm = aftm / aitm

    return cftm, aftm


def substrate_thermoelastic(f, materials, wBeam):
    """Substrate thermal displacement noise spectrum from thermoelastic fluctuations

    :f: frequency array in Hz
    :materials: gwinc optic materials structure
    :wBeam: beam radius (at 1 / e^2 power)

    :returns: displacement noise power spectrum at :f:, in meters

    """
    sigma = materials.Substrate.MirrorSigma
    rho = materials.Substrate.MassDensity
    kappa = materials.Substrate.MassKappa # thermal conductivity
    alpha = materials.Substrate.MassAlpha # thermal expansion
    CM = materials.Substrate.MassCM # heat capacity @ constant mass
    Temp = materials.Substrate.Temp # temperature
    kBT = const.kB * materials.Substrate.Temp

    S = 8*(1+sigma)**2*kappa*alpha**2*Temp*kBT # note kBT has factor Temp
    S /= (sqrt(2*pi)*(CM*rho)**2)
    S /= (wBeam/sqrt(2))**3 # LT 18 less factor 1/omega^2

    # Corrections for finite test masses:
    S *= substrate_thermoelastic_FiniteCorr(materials, wBeam)

    return S/(2*pi*f)**2


def substrate_thermoelastic_FiniteCorr(materials, wBeam):
    """Substrate thermoelastic noise finite-size test mass correction

    :materials: gwinc optic materials structure
    :wBeam: beam radius (at 1 / e^2 power)

    :returns: correction factor

    (Liu & Thorne gr-qc/0002055 equation 46)

    Equation references to Bondu, et al. Physics Letters A 246 (1998)
    227-236 (hereafter BHV) or Liu and Thorne gr-qc/0002055 (hereafter LT)

    """
    a = materials.MassRadius
    h = materials.MassThickness
    sigma = materials.Substrate.MirrorSigma

    # LT uses power e-folding
    r0 = wBeam/sqrt(2)
    km = zeta/a

    Qm = exp(-2*km*h) # LT 35a

    pm = exp(-(km*r0)**2/4)/(pi*(a*j0m)**2) # LT 37

    c0 = 6*(a/h)**2*sum(j0m*pm/zeta**2) # LT 32
    c1 = -2*c0/h # LT 32
    p0 = 1/(pi*a**2) # LT 28
    c1 += p0/(2*h) # LT 40

    coeff = (1-Qm)*((1-Qm)*(1+Qm)+8*h*km*Qm)
    coeff += 4*(h*km)**2*Qm*(1+Qm)
    coeff *= km*(pm*j0m)**2*(1-Qm)
    coeff /= ((1-Qm)**2-4*(h*km)**2*Qm)**2
    coeff = sum(coeff) + h*c1**2/(1+sigma)**2
    coeff *= (sqrt(2*pi)*r0)**3*a**2 # LT 46

    return coeff
