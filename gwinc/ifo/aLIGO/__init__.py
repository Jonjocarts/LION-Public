from gwinc.ifo.noises import *
from gwinc.ifo import PLOT_STYLE


class aLIGO(nb.Budget):

    name = 'Advanced LIGO'

    noises = [
        QuantumVacuum,
        Seismic,
        Newtonian,
        SuspensionThermal,
        CoatingBrownian,
        CoatingThermoOptic,
        SubstrateBrownian,
        SubstrateThermoElastic,
        ExcessGas,
    ]

    calibrations = [
        Strain,
    ]

    plot_style = PLOT_STYLE
