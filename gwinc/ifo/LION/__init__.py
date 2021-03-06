from gwinc.ifo.noises import *
from gwinc.ifo import PLOT_STYLE


class Newtonian(nb.Budget):
    """Newtonian Gravity

    """

    name = 'Newtonian'

    style = dict(
        label='Newtonian',
        color='#15b01a',
    )

    noises = [
        NewtonianRayleigh,
        NewtonianBody,
        NewtonianInfrasound,
    ]


class Coating(nb.Budget):
    """Coating Thermal

    """

    name = 'Coating'

    style = dict(
        label='Coating Thermal',
        color='#fe0002',
    )

    noises = [
        CoatingBrownian,
        CoatingThermoOptic,
    ]


class Substrate(nb.Budget):
    """Substrate Thermal

    """

    name = 'Substrate'

    style = dict(
        label='Substrate Thermal',
        color='#fb7d07',
        linestyle='--',
    )

    noises = [
        ITMThermoRefractive,
        SubstrateBrownian,
        SubstrateThermoElastic,
    ]


class LION(nb.Budget):

    name = 'LION'

    noises = [
        QuantumVacuum,
        Seismic,
        SuspensionThermal,
        Coating,
        Substrate,
    ]
#        Newtonian, ExcessGas,

    calibrations = [
        Strain,
    ]

    plot_style = PLOT_STYLE
