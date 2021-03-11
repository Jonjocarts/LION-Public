from __future__ import division
import os
import sys
import logging
import importlib
import numpy as np

from .ifo import IFOS
from .struct import Struct
from .plot import plot_noise


logger = logging.getLogger('gwinc')


def _load_module(name_or_path):
    """Load module from name or path.

    Return loaded module and module path.

    """
    if os.path.exists(name_or_path):
        path = name_or_path.rstrip('/')
        modname = os.path.splitext(os.path.basename(path))[0]
        if os.path.isdir(path):
            path = os.path.join(path, '__init__.py')
        spec = importlib.util.spec_from_file_location(modname, path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[mod.__name__] = mod
        spec.loader.exec_module(mod)
    else:
        mod = importlib.import_module(name_or_path)
    try:
        path = mod.__path__[0]
    except AttributeError:
        path = mod.__file__
    return mod, path


def load_budget(name_or_path, freq=None):
    """Load GWINC Budget

    Accepts either the name of a built-in canonical budget (see
    gwinc.IFOS), the path to a budget package (directory) or module
    (ending in .py), or the path to an IFO struct definition file (see
    gwinc.Struct).

    If the budget is a package directory which includes an 'ifo.yaml'
    file the ifo Struct will be loaded from that file and assigned to
    the budget.ifo attribute.  If a struct definition file is provided
    the base aLIGO budget definition and ifo will be assumed.

    Returns a Budget object instantiated with the provided frequency
    array (if specified), and with any included ifo assigned as an
    object attribute.  If a frequency array is not provided and the
    budget class does not define it's own, the frequency array must be
    provided at budget update() or run() time.

    """
    ifo = None

    if os.path.exists(name_or_path):
        path = name_or_path.rstrip('/')
        bname, ext = os.path.splitext(os.path.basename(path))

        if ext in Struct.STRUCT_EXT:
            logger.info("loading struct {}...".format(path))
            ifo = Struct.from_file(path)
            bname = 'aLIGO'
            modname = 'gwinc.ifo.aLIGO'

        else:
            modname = path

    else:
        if name_or_path not in IFOS:
            raise RuntimeError("Unknonw IFO '{}' (available IFOs: {}).".format(
                name_or_path,
                IFOS,
            ))
        bname = name_or_path
        modname = 'gwinc.ifo.'+name_or_path

    logger.info("loading module {}...".format(modname))
    mod, modpath = _load_module(modname)
    Budget = getattr(mod, bname)
    ifopath = os.path.join(modpath, 'ifo.yaml')
    if not ifo and os.path.exists(ifopath):
        ifo = Struct.from_file(ifopath)
    return Budget(freq=freq, ifo=ifo)


def gwinc(freq, ifo, source=None, plot=False, PRfixed=True):
    """Calculate strain noise budget for a specified interferometer model.

    Argument `freq` is the frequency array for which the noises will
    be calculated, and `ifo` is the IFO model (see the `load_budget()`
    function).  The nominal 'aLIGO' budget structure will be used.

    If `source` structure provided, so evaluates the sensitivity of
    the detector to several potential gravitational wave
    sources.

    If `plot` is True a plot of the budget will be created.

    Returns tuple of (score, noises, ifo)

    """
    # assume generic aLIGO configuration
    # FIXME: how do we allow adding arbitrary addtional noise sources
    # from just ifo description, without having to specify full budget
    budget = load_budget('aLIGO', freq)
    traces = budget.run()
    plot_style = getattr(budget, 'plot_style', {})

    # construct matgwinc-compatible noises structure
    noises = {}
    for name, (data, style) in traces.items():
        noises[style.get('label', name)] = data
    noises['Freq'] = freq

    pbs = ifo.gwinc.pbs
    parm = ifo.gwinc.parm
    finesse = ifo.gwinc.finesse
    prfactor = ifo.gwinc.prfactor
    if ifo.Laser.Power * prfactor != pbs:
        logger.warning("Thermal lensing limits input power to {} W".format(pbs/prfactor))

    # report astrophysical scores if so desired
    score = None
    if source:
        logger.warning("Source score calculation currently not supported.  See `inspiral-range` package for similar functionality:")
        logger.warning("https://git.ligo.org/gwinc/inspiral-range")
        # score = int731(freq, noises['Total'], ifo, source)
        # score.Omega = intStoch(freq, noises['Total'], 0, ifo, source)

    # --------------------------------------------------------
    # output graphics

    if plot:
        logger.info('Laser Power:            %7.2f Watt' % ifo.Laser.Power)
        logger.info('SRM Detuning:           %7.2f degree' % (ifo.Optics.SRM.Tunephase*180/np.pi))
        logger.info('SRM transmission:       %9.4f' % ifo.Optics.SRM.Transmittance)
        logger.info('ITM transmission:       %9.4f' % ifo.Optics.ITM.Transmittance)
        logger.info('PRM transmission:       %9.4f' % ifo.Optics.PRM.Transmittance)
        logger.info('Finesse:                %7.2f' % finesse)
        logger.info('Power Recycling Gain:   %7.2f' % prfactor)
        logger.info('Arm Power:              %7.2f kW' % (parm/1000))
        logger.info('Power on BS:            %7.2f W' % pbs)

        # coating and substrate thermal load on the ITM
        PowAbsITM = (pbs/2) * \
                    np.hstack([(finesse*2/np.pi) * ifo.Optics.ITM.CoatingAbsorption,
                               (2 * ifo.Materials.MassThickness) * ifo.Optics.ITM.SubstrateAbsorption])

        logger.info('Thermal load on ITM:    %8.3f W' % sum(PowAbsITM))
        logger.info('Thermal load on BS:     %8.3f W' %
                     (ifo.Materials.MassThickness*ifo.Optics.SubstrateAbsorption*pbs))
        if (ifo.Laser.Power*prfactor != pbs):
            logger.info('Lensing limited input power: %7.2f W' % (pbs/prfactor))

        if score:
            logger.info('BNS Inspiral Range:     ' + str(score.effr0ns) + ' Mpc/ z = ' + str(score.zHorizonNS))
            logger.info('BBH Inspiral Range:     ' + str(score.effr0bh) + ' Mpc/ z = ' + str(score.zHorizonBH))
            logger.info('Stochastic Omega: %4.1g Universes' % score.Omega)

        plot_noise(ifo, traces, **plot_style)

    return score, noises, ifo
