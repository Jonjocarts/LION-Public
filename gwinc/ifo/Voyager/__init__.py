from gwinc.ifo.noises import *
from gwinc.ifo import PLOT_STYLE


class Voyager(nb.Budget):

    name = 'Voyager'

    noises = [
        QuantumVacuum,
        Seismic,
        Newtonian,
        SuspensionThermal,
        CoatingBrownian,
        CoatingThermoOptic,
        ITMThermoRefractive,
        SubstrateBrownian,
        SubstrateThermoElastic,
        ExcessGas,
    ]

    calibrations = [
        Strain,
    ]

    plot_style = PLOT_STYLE
