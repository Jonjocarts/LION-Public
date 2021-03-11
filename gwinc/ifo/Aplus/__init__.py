from gwinc.ifo.noises import *
from gwinc.ifo import PLOT_STYLE


class Aplus(nb.Budget):

    name = 'A+'

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
