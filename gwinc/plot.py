from numpy import sqrt
from collections.abc import Mapping


def plot_noise(
        freq,
        traces,
        ax=None,
        **kwargs
):
    """Plot a GWINC noise budget from calculated noises

    If an axis handle is provided it will be used for the plot.

    Returns the figure handle.

    """
    if ax is None:
        import matplotlib.pyplot as plt
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
    else:
        fig = ax.figure

    ylim = kwargs.get('ylim')

    for name, trace in traces.items():
        if isinstance(trace, Mapping):
            trace = trace['Total']

        try:
            data = trace[0]
            style = dict(**trace[1])
        except TypeError:
            data = trace
            style = {}
        # assuming all data is PSD
        data = sqrt(data)
        if name == 'Total' and not style:
            style = dict(
                color='#000000',
                alpha=0.6,
                lw=4,
            )
            if ylim is None:
                ylim = [1e-25, 1e-18]
        if 'label' not in style:
            style['label'] = name
        if 'linewidth' in style:
            style['lw'] = style['linewidth']
        elif 'lw' not in style:
            style['lw'] = 3

        ax.loglog(freq, data, **style)

    ax.grid(
        True,
        which='both',
        lw=0.5,
        ls='-',
        alpha=0.5,
    )

    ax.legend(
        ncol=2,
        fontsize='small',
    )

    ax.autoscale(enable=True, axis='y', tight=True)
    if ylim:
        ax.set_ylim(ylim)
    ax.set_xlim(freq[0], freq[-1])

    ax.set_xlabel('Frequency [Hz]')
    if 'ylabel' in kwargs:
        ax.set_ylabel(kwargs['ylabel'])
    if 'title' in kwargs:
        ax.set_title(kwargs['title'])

    return fig
