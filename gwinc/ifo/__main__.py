import argparse
import numpy as np
import matplotlib.pyplot as plt

from . import IFOS, PLOT_STYLE
from .. import load_budget


FLO = 3
FHI = 10000
NPOINTS = 3000
YLIM = (1e-25, 1e-20)


def main():
    parser = argparse.ArgumentParser(
        description="Reference IFO comparison plot",
    )
    parser.add_argument(
        '--save', '-s',
        help="save plot to file (.pdf/.png/.svg)")
    args = parser.parse_args()

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    freq = np.logspace(np.log10(FLO), np.log10(FHI), NPOINTS)

    budgets = {}
    range_pad = 0
    for ifo in IFOS:
        budget = load_budget(ifo, freq)
        name = budget.name
        budgets[name] = budget
        range_pad = max(len(name), range_pad)

    for name, budget in budgets.items():
        data = budget.calc()
        try:
            import inspiral_range
            label_range = ' {:>6.0f} Mpc'.format(
                inspiral_range.range(freq, data),
            )
        except ModuleNotFoundError:
            label_range = ''

        label = '{name:<{pad}}{range}'.format(
            name=name,
            pad=range_pad,
            range=label_range,
        )
        ax.loglog(freq, np.sqrt(data), label=label, lw=2)

    ax.grid(
        True,
        which='both',
        lw=0.5,
        ls='-',
        alpha=0.5,
    )

    ax.legend(
        fontsize='small',
        prop={'family': 'monospace'},
    )

    ax.autoscale(enable=True, axis='y', tight=True)
    ax.set_ylim(*YLIM)
    ax.set_xlim(freq[0], freq[-1])

    ax.set_xlabel('Frequency [Hz]')
    ax.set_ylabel(PLOT_STYLE['ylabel'])
    ax.set_title("PyGWINC reference IFO strain comparison")

    if args.save:
        fig.savefig(args.save)
    else:
        plt.show()


if __name__ == '__main__':
    main()
