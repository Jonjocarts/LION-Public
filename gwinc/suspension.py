from __future__ import division
from numpy import pi, sqrt, sin, cos, tan, real, imag
import numpy as np

from . import logger
from .const import g


# supported fiber geometries
FIBER_TYPES = [
    'Round',
    'Ribbon',
    'Tapered',
]


def generate_symbolic_tfs(stages=4):
    import sympy as sp

    # construct quad pendulum equation of motion matrix
    ksyms = sp.numbered_symbols('k')
    msyms = sp.numbered_symbols('m')
    w = sp.symbols('w')
    k = [next(ksyms) for n in range(stages)]
    m = [next(msyms) for n in range(stages)]
    A = sp.zeros(stages)
    for n in range(stages-1):
        # mass and restoring forces (diagonal elements)
        A[n, n] = k[n] + k[n+1] - m[n] * w**2
        # couplings to stages above and below
        A[n, n+1] = -k[n+1]
        A[n+1, n] = -k[n+1]
    # mass and restoring force of bottom stage
    A[-1, -1] = k[-1] - m[-1] * w**2

    # want TM equations of motion, so index 4
    b = sp.zeros(stages, 1)
    b[-1] = 1

    # solve linear system
    xsyms = sp.numbered_symbols('x')
    x = [next(xsyms) for n in range(stages)]
    ans = sp.linsolve((A, b), x)
    return ans


def tst_force_to_tst_displ(k, m, f):
    """transfer function for quad pendulum

    """
    k0, k1, k2, k3 = k
    m0, m1, m2, m3 = m
    w = 2*pi*f
    X3 = (k2**2*(k0 + k1 - m0*w**2) + (k1**2 - (k0 + k1 - m0*w**2)*(k1 + k2 - m1*w**2))*(k2 + k3 - m2*w**2))/(-k3**2*(k1**2 - (k0 + k1 - m0*w**2)*(k1 + k2 - m1*w**2)) + (k3 - m3*w**2)*(k2**2*(k0 + k1 - m0*w**2) - (-k1**2 + (k0 + k1 - m0*w**2)*(k1 + k2 - m1*w**2))*(k2 + k3 - m2*w**2)))
    return X3


def top_displ_to_tst_displ(k, m, f):
    """transfer function for quad pendulum

    """
    k0, k1, k2, k3 = k
    m0, m1, m2, m3 = m
    w = 2*pi*f
    X0 = k1*k2*k3/(k3**2*(k1**2 - (k0 + k1 - m0*w**2)*(k1 + k2 - m1*w**2)) - (k3 - m3*w**2)*(k2**2*(k0 + k1 - m0*w**2) + (k1**2 - (k0 + k1 - m0*w**2)*(k1 + k2 - m1*w**2))*(k2 + k3 - m2*w**2)))
    return X0 * k0


def getJointParams(sus, n):
    """Return relevant material properties from SUS struct

    Parameters are returned as a pair of tuples,
    (isLower, Temperature, WireMaterialStruct, BladeMaterialStruct),
    corresponding to the upper and lower joints of stage n.

    """
    # Here the suspension stage list is reversed so that the last stage
    # in the list is the test mass
    stages = sus.Stage[::-1]
    last_stage = len(stages) - 1
    stage = stages[n]
    if 'Temp' in stage:
        Temp = stage.Temp
    else:
        Temp = sus.Temp

    TempLower = Temp
    if n == 0:
        # For the top stage: if sus.Temp exists, interpret it as the
        # upper joint temperature.  Otherwise, assume upper and lower
        # joints of the top stage have the same temperature.
        if 'Temp' in sus:
            TempUpper = sus.Temp
        else:
            TempUpper = Temp
    else:
        # Below the top stage: assume that the upper joint has the same
        # temperature as the nearby lower joint of the preceding stage.
        if 'Temp' in stages[n-1]:
            TempUpper = stages[n-1].Temp
        else:
            TempUpper = sus.Temp

    ##############################
    # material parameters

    # support wire
    # upper and lower joint material properties may be different
    # due to temperature gradient
    if 'WireMaterialUpper' in stage and 'WireMaterialLower' in stage:
        wireMatUpper = stage.WireMaterialUpper
        wireMatLower = stage.WireMaterialLower
    elif 'WireMaterial' in stage:
        wireMatUpper = stage.WireMaterial
        wireMatLower = stage.WireMaterial
    elif n == last_stage:
        wireMatUpper = 'Silica'
        wireMatLower = 'Silica'
    else:
        wireMatUpper = 'C70Steel'
        wireMatLower = 'C70Steel'

    WireMaterialUpper = sus[wireMatUpper]
    WireMaterialLower = sus[wireMatLower]
    logger.debug('stage {} wires: {}, {}'.format(n, wireMatUpper, wireMatLower))

    # support blade (upper joint only)
    if 'BladeMaterial' in stage:
        bladeMat = stage.BladeMaterial
    else:
        bladeMat = 'MaragingSteel'

    BladeMaterial = sus[bladeMat]

    logger.debug('stage {} blades: {}'.format(n, bladeMat))

    ULparams = ((0, TempUpper, WireMaterialUpper, BladeMaterial),
                (1, TempLower, WireMaterialLower, None))
    return ULparams


def wireGeometry(r, N, RibbonThickness=None, TaperedEndRadius=None, **kwargs):
    """Compute wire geometry-dependent factors

    r is the wire radius, or ribbon width.
    N is the number of wires or ribbons
    RibbonThickness must be set when ribbons are used, or
    TaperedEndRadius when tapered fibers are used.
    Other kwargs are ignored.

    Returns cross-sectional areas (central and end) and moment of inertia,
    and modified surface to volume ratios (vertical and horizontal).

    """
    # Usual case: round wire/fiber
    xsect = pi * r**2  # cross-sectional area
    xsectEnd = xsect   # only differs for tapered fiber
    # cross-sectional moment of inertia
    # Ref: Gretarsson et al, Phys. Lett. A 270 (2000) 108-114 Eq.(2)
    xII = r**4 * pi / 4  # x-sectional moment of inertia
    # surface to volume ratio, vertical
    # Surface = 2*pi*r*h
    # Volume  = pi*r^2*h
    # Geometrical factor \mu is unity (corresponding to uniform strain energy)
    mu_v = 2 / r
    # surface to volume ratio, horizontal
    # Ref: Gretarsson and Harry, Rev. Sci. Inst. 70 (1999) 4081-4087
    # Appendix (A8)
    # There is a geometrical factor \mu to calculate the ratio of the
    # energies between the ones stored in surface and the bulk.
    # For a transversely oscillating fiber of circular cross section, this
    # factor is \mu = 2 as shown in (A11)
    mu_h = mu_v * 2

    # Special case: ribbon
    if RibbonThickness is not None:
        W = r
        t = RibbonThickness
        xsect = W * t        # cross-sectional area
        xsectEnd = xsect     # only differs for tapered fiber
        xII = (W * t**3)/12  # x-sectional moment of inertia
        # surface to volume ratio, vertical
        # Surface = 2*(W+t)*h
        # Volume  = W*t*h
        mu_v = 2 * (W + t) / (W * t)
        # surface to volume ratio, horizontal
        # Ref: Gretarsson et al, Phys. Lett. A 270 (2000) 108-114
        # There is a geometrical factor \mu (same as above).
        # This factor is shown in Eq.(6)
        mu_h = mu_v * (3 * N * W + t) / (N * W + t)

    # Special case: tapered
    elif TaperedEndRadius is not None:
        r_end = TaperedEndRadius
        xsectEnd = pi * r_end**2  # cross-sectional area (for delta_h)
        xII = pi * r_end**4 / 4   # x-sectional moment of inertia
        mu_h = 4 / r_end          # surface to volume ratio, horizontal
        # mu_v is unchanged (assumes the middle part of the fiber dominates)

    return (xsect, xsectEnd, xII, mu_v, mu_h)


def wireTELoss(w, tension, xsectEnd, xII, Temp, alpha, beta, rho, C, K, Y,
               RibbonThickness=None, **kwargs):
    """Thermoelastic calculations for wires

    Repeated for upper and lower joint of each stage.
    w = angular frequency
    tension = weight supported per wire
    xsectEnd = cross sectional area of wire end
    xII = cross sectional moment of inertia
    Temp = temperature
    alpha = coeff of thermal expansion
    beta = temp dependence of Young's modulus
    rho = mass density
    C = heat capacity
    K = thermal conductivity W/(m K)
    Y = Young's modulus
    RibbonThickness must be set when ribbons are used
    Other kwargs are ignored

    Returns the loss angle associated with thermoelastic damping
    (wire horizontal)

    """
    # horizontal TE time constant, wires
    # The constant 7.37e-2 is 1/(4*q0^2) from eq 12, C. Zener 10.1103/PhysRev.53.90 (1938)
    tau = 7.37e-2 * 4 * (rho * C * xsectEnd) / (pi * K)

    # deal with ribbon geometry
    if RibbonThickness is not None:
        t = RibbonThickness
        tau = (rho * C * t**2) / (K * pi**2)

    # delta: TE factor
    # The first term expresses the cancellation of the CTE and dY/dT
    # beta = (dY/dT)/Y
    # Refs:
    # Cagnoli G and Willems P 2002 Phys. Rev. B 65 174111
    # Cumming A V et al 2009 Class. Quantum Grav. 26 215012
    # horizontal delta, wires
    delta = (alpha - tension * beta / (xsectEnd * Y))**2 * Y * Temp / (rho * C)

    phi_TE = delta * tau * w / (1 + w**2 * tau**2)
    return phi_TE


def bladeTELoss(w, t, Temp, alpha, beta, rho, C, K, Y):
    """Thermoelastic calculations for blades

    Invoked for upper joint only (there is no lower blade)
    w = angular frequency
    t = blade thickness
    Temp = temperature
    alpha = coeff of thermal expansion
    beta = temp dependence of Young's modulus
    rho = mass density
    C = heat capacity
    K = thermal conductivity W/(m K)
    Y = Young's modulus

    Returns the loss angle associated with thermoelastic damping
    (blade vertical)

    """
    # vertical TE time constant, blades
    tau = (rho * C * t**2) / (K * pi**2)

    # vertical delta, blades
    # Here the TE cancellation is ignored
    delta = Y * alpha**2 * Temp / (rho * C)

    phi_TE = delta * tau * w / (1 + w**2 * tau**2)
    return phi_TE


##############################
# Ref "GG"
#   Suspensions thermal noise in the LIGO gravitational wave detector
#   Gabriela Gonzalez, https://doi.org/10.1088/0264-9381/17/21/305
# Ref "GS"
#   Brownian motion of a mass suspended by an anelastic wire
#   Gonzalez & Saulson, https://doi.org/10.1121/1.410467
#
# Note:
#  I in GG = xII
#  rho in GG = rho * xsect
#  delta in GG = d_bend
#  T in GG = tension
##############################

def continuumWireKh(w, N, length, tension, xsect, xII, rho, Y, phi):
    """Horizontal spring constant, including violin modes

    w = angular frequency
    N = number of wires
    length = wire length
    tension = weight supported per wire
    xsect = wire cross sectional area
    xII = cross sectional moment of inertia
    rho = mass density
    Y = Young's modulus
    phi = loss angle

    Returns the spring constant (wire horizontal)
    """
    Y = Y * (1 + 1j * phi)  # complex Young's modulus

    # simplification factors for later calculations
    # Note: a previous version of this suspension thermal noise
    # calculation used a factor "simp3" which was an incorrect
    # approximation for k_e (see GS eq 10).  Here we approximate k_e
    # as 1/delta.  This is equivalent to assuming delta << 1/k,
    # i.e. the bending length of the fiber is much shorter than the
    # 'ideal string' wavelength.
    k = sqrt(rho * xsect / tension) * w
    d_bend = sqrt(Y * xII / tension)  # complex d_bend
    dk = k * d_bend                   # dk << 1 was assumed
    coskl = cos(k * length)
    sinkl = sin(k * length)

    # numerator, horiz spring constant
    #   numerator of K_xx in eq 9 of GG
    #     = T k (cos(k L) + k delta sin(k L))
    #   for w -> 0, this reduces to N_w * T * k
    khnum = N * tension * k * (coskl + dk * sinkl)

    # denominator, horiz spring constant
    #   D after equation 8 in GG
    #   D = sin(k L) - 2 k delta cos(k L)
    #   for w -> 0, this reduces to k (L - 2 delta)
    khden = sinkl - 2 * dk * coskl

    return khnum/khden


def continuumWireKv(w, N, length, xsect, xsectEnd, rho, Y, phi,
                    TaperedEndLength=None, **kwargs):
    """Vertical spring constant, including bounce mode.

    w = angular frequency
    N = number of wires
    length = wire length
    xsect = wire cross sectional area
    xsectEnd = cross sectional area of wire end
    rho = mass density
    Y = Young's modulus
    phi = loss angle
    TaperedEndLength must be set when tapered fibers are used
    Other kwargs are ignored

    Returns the spring constant (wire vertical)
    """
    Y = Y * (1 + 1j * phi)  # complex Young's modulus
    k = sqrt(rho / Y) * w

    kv = N * xsect * Y * k / tan(k * length)
    # deal with tapered geometry
    if TaperedEndLength is not None:
        l_end = TaperedEndLength
        l_mid = length - 2*l_end
        kv_mid = N * xsect * Y * k / tan(k * l_mid)
        kv_end = N * xsectEnd * Y * k / tan(k * l_end)
        kv = 1/(2/kv_end + 1/kv_mid)

    return kv


def suspQuad(f, sus):
    """Suspension for quadruple pendulum

    `f` is frequency vector, `sus` is suspension model.
    Violin modes are included for the bottom stage.

      sus.FiberType should be: 0=round, 1=ribbons.

    hForce, vForce = transfer functions from the force on the TM to TM
    motion.  These should have the correct losses for the mechanical
    system such that the thermal noise is:

    dxdF = force on TM along beam line to position of TM along beam line
         = hForce + theta^2 * vForce
         = admittance / (i * w)

    where theta = sus.VHCoupling.theta.

    Since this is just suspension thermal noise, the TM internal modes
    and coating properties should not be included.

    hTable, vTable = TFs from support motion to TM motion

    Ah = horizontal equations of motion
    Av = vertical equations of motion

    Adapted from code by Morag Casey (Matlab) and Geppo Cagnoli
    (Maple).  Modification for the different temperatures between the
    stages by K.Arai.

    """
    w = 2 * pi * f

    # NOTE: For historical reasons, the IFO struct numbers stages from the
    # bottom up (0 is the TM), while this code numbers them from the top down
    # (3 is the TM).  Here the suspension stage list is reversed so that the
    # last stage in the list is the test mass
    stages = sus.Stage[::-1]
    last_stage = len(stages) - 1

    # bottom stage fiber type
    FiberType = sus.FiberType
    assert FiberType in FIBER_TYPES

    ##############################
    # Compute cumulative weight of suspension
    masses = np.array([stage.Mass for stage in stages])
    # weight support by lower stages
    Mgs = g * np.flipud(np.cumsum(np.flipud(masses)))

    ##############################
    # Complex spring constants
    khr = np.zeros([len(stages), len(w)])
    khi = np.zeros([len(stages), 2, len(w)])
    kvr = np.zeros([len(stages), len(w)])
    kvi = np.zeros([len(stages), 2, len(w)])

    for n, stage in enumerate(stages):

        ##############################
        # main suspension parameters
        Mg = Mgs[n]
        length = stage.Length

        # Correction for the pendulum restoring force
        kh0 = Mg / length   # N/m, horiz. spring constant, stage n

        kv0_blade = stage.K    # N/m, vert. spring constant (from blade)

        if not np.isnan(stage.WireRadius):
            r_w = stage.WireRadius
        elif FiberType == 'Ribbon':
            r_w = sus.Ribbon.Width
        else:
            r_w = sus.Fiber.Radius

        # blade thickness
        t_b = stage.Blade
        # number of support wires
        N_w = stage.NWires
        tension = Mg / N_w

        # wire geometry
        wireShape = {}
        if n == last_stage and FiberType == 'Ribbon':
            wireShape['RibbonThickness'] = sus.Ribbon.Thickness
        elif n == last_stage and FiberType == 'Tapered':
            wireShape['TaperedEndRadius'] = sus.Fiber.EndRadius
            wireShape['TaperedEndLength'] = sus.Fiber.EndLength
        xsect, xsectEnd, xII, mu_v, mu_h = wireGeometry(r_w, N_w, **wireShape)

        jointParams = getJointParams(sus, n)
        for (isLower, Temp, wireMat, bladeMat) in jointParams:
            # wire parameters
            alpha_w = wireMat.Alpha  # coeff of thermal expansion
            beta_w = wireMat.dlnEdT  # temp dependence of Young's modulus
            rho_w = wireMat.Rho      # mass density
            C_w = wireMat.C          # heat capacity
            K_w = wireMat.K          # thermal conductivity W/(m K)
            Y_w = wireMat.Y          # Young's modulus
            phi_w = wireMat.Phi      # loss angle

            # surface loss dissipation depth
            # About the surface loss depth see Ref:
            # A. M. Gretarsson and G. M. Harry
            # Rev. Sci. Instrum. 70 (1999) P.4081-4087
            if 'Dissdepth' in wireMat:
                ds_w = wireMat.Dissdepth
            else:
                ds_w = 0             # ignore surface effects

            if not isLower:  # blade parameters (there is no lower blade)
                alpha_b = bladeMat.Alpha # coeff of thermal expansion
                beta_b = bladeMat.dlnEdT # temp dependence of Young's modulus
                rho_b = bladeMat.Rho     # mass density
                C_b = bladeMat.C         # heat capacity
                K_b = bladeMat.K         # thermal conductivity W/(m K)
                Y_b = bladeMat.Y         # Young's modulus
                phi_b = bladeMat.Phi     # loss angle

            # bending length, and dilution factors
            if 'Dilution' in stage and not np.isnan(stage.Dilution):
                # if a specific dilution factor is given, use it
                dil = stage.Dilution
            else:
                # if not, calculate
                # This formula is for the effective dilution in the GW band
                # (as discussed in ref GG section 2.5)
                d_bend = sqrt(Y_w * xII / tension)
                dil = length / d_bend

            ##############################
            # Loss Calculations for wires and blades
            # Track losses from the upper and lower wire joints separately
            # The factor of 2 comes from that the loss was calculated
            # for both upper and lower joints
            # Blades are always attached at the top, there is no "lower" blade
            # The final stage is handled specially and the spring constant
            # calculation below is not used

            # horizontal loss factor for round wires
            # phi_w: bulk loss (brownian motion)
            # (mu_h*ds_w): surface loss (brownian motion)
            # nominally this term becomes zero for steel wires as ds_w is zero
            # The last term is for TE loss
            phih_TE = wireTELoss(w, tension, xsectEnd, xII, Temp,
                                 alpha_w, beta_w, rho_w, C_w, K_w, Y_w,
                                 **wireShape)
            phih = phi_w * (1 + mu_h * ds_w) + phih_TE

            # complex spring constant, horizontal
            khr[n, :] = kh0
            khi[n, isLower, :] = kh0 * phih / dil / 2

            if not isLower:
                # vertical loss factor, blades
                # The bulk loss term and the TE loss term
                # wire loss and stiffness are neglected here
                phiv_TE = bladeTELoss(w, t_b, Temp,
                                      alpha_b, beta_b, rho_b, C_b, K_b, Y_b)
                phiv_blade = phi_b + phiv_TE

                # complex spring constant, vertical
                kvr[n, :] = kv0_blade
                kvi[n, 0, :] = kv0_blade * phiv_blade


        if n == last_stage:
            # loss factor, vertical (no blades)
            # wire vertical thermoelastic loss is negligible (Saulson 1990)
            phiv = phi_w * (1 + mu_v * ds_w)

            kv4 = continuumWireKv(w, N_w, length, xsect, xsectEnd,
                                  rho_w, Y_w, phiv, **wireShape)
            kh4 = continuumWireKh(w, N_w, length, tension, xsect, xII,
                                  rho_w, Y_w, phih)

            # if blades present, combine two vertical springs
            if not np.isnan(kv0_blade):
                kv4wire = kv4
                kv4blade = kvr[n, :] + 1j*kvi[n, 0, :]
                kv4 = 1/(1/kv4wire + 1/kv4blade)

            kvr[n, :] = real(kv4)
            kvi[n, 0, :] = imag(kv4)
            kvi[n, 1, :] = 0
            khr[n, :] = real(kh4)
            khi[n, 0, :] = imag(kh4)
            khi[n, 1, :] = 0


    ###############################################################
    # Equations of motion for the system
    ###############################################################

    # Calculate horizontal/vertical TFs turning on the loss of each
    # joint one by one, for the thermal noise calculation
    hForce = np.zeros([2*len(stages), len(w)], dtype=complex)
    vForce = np.zeros([2*len(stages), len(w)], dtype=complex)

    for m in range(2*len(stages)):
        # turn on only the loss of the current joint
        n = int(m/2)  # stage number
        isLower = m % 2
        stage_selection = np.zeros((len(stages), 1))
        stage_selection[n] = 1

        # horizontal
        # only the imaginary part due to the specified joint is used.
        k = khr + 1j*khi[:,isLower,:]*stage_selection
        # calculate TFs
        hForce[m,:] = tst_force_to_tst_displ(k, masses, f)

        # vertical
        # only the imaginary part due to the specified joint is used
        k = kvr + 1j*kvi[:,isLower,:]*stage_selection
        # calculate TFs
        vForce[m,:] = tst_force_to_tst_displ(k, masses, f)

    # Calculate horizontal/vertical TFs with all losses on,
    # for the vibration isolation calculation
    k = khr + 1j*(khi[:,0,:] + khi[:,1,:])
    hTable = top_displ_to_tst_displ(k, masses, f)
    k = kvr + 1j*(kvi[:,0,:] + kvi[:,1,:])
    vTable = top_displ_to_tst_displ(k, masses, f)

    return hForce, vForce, hTable, vTable
