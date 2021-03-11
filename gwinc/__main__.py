from __future__ import print_function
import os
import signal
import logging
import argparse
import numpy as np

from . import IFOS, load_budget, plot_noise, logger

logger.setLevel(os.getenv('LOG_LEVEL', 'WARNING').upper())
formatter = logging.Formatter('%(message)s')
handler = logging.StreamHandler()
handler.setFormatter(formatter)
logger.addHandler(handler)

##################################################

description = """GWINC noise budget tool

IFOs can be specified by name of included canonical budget (see
below), or by path to a budget module (.py), description file
(.yaml/.mat/.m), or HDF5 data file (.hdf5/.h5).  Available included
IFOs are:

  {}

""".format(', '.join(["{}".format(ifo) for ifo in IFOS]))
# for ifo in available_ifos():
#     description += "  '{}'\n".format(ifo)
description += """

By default the noise budget of the specified IFO will be loaded and
plotted with an interactive plotter.  Individual IFO parameters can be
overriden with the --ifo option:

  gwinc --ifo Optics.SRM.Tunephase=3.14 ...

If the --save option is specified the plot will be saved directly to a
file (without display) (various file formats are supported, indicated
by file extension).  If the requested extension is 'hdf5' or 'h5' then
the noise traces and IFO parameters will be saved to an HDF5 file.

If the inspiral_range package is available, various figures of merit
can be calculated for the resultant spectrum with the --fom option,
e.g.:

  gwinc --fom horizon ...
  gwinc --fom range:m1=20,m2=20 ...

See the inspiral_range package documentation for details.
"""

IFO = 'aLIGO'
FREQ = '5:3000:6000'
FOM = 'range:m1=1.4,m2=1.4'
DATA_SAVE_FORMATS = ['.hdf5', '.h5']

parser = argparse.ArgumentParser(
    prog='gwinc',
    description=description,
    formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument(
    '--freq', '-f', metavar='FLO:[NPOINTS:]FHI',
    help="logarithmic frequency array specification in Hz [{}]".format(FREQ))
parser.add_argument(
    '--ifo', '-o', metavar='PARAM=VAL',
    #nargs='+', action='extend',
    action='append',
    help="override budget IFO parameter (may be specified multiple times)")
parser.add_argument(
    '--title', '-t',
    help="plot title")
parser.add_argument(
    '--fom', metavar='FUNC[:PARAM=VAL,...]', default=FOM,
    help="use inspiral_range.FUNC to calculate range figure-of-merit on resultant spectrum")
group = parser.add_mutually_exclusive_group()
group.add_argument(
    '--interactive', '-i', action='store_true',
    help="launch interactive shell after budget processing")
group.add_argument(
    '--save', '-s', metavar='PATH', action='append',
    help="save plot (.png/.pdf/.svg) or budget traces (.hdf5/.h5) to file (may be specified multiple times)")
group.add_argument(
    '--yaml', '-y', action='store_true',
    help="print IFO as yaml to stdout and exit (budget not calculated)")
group.add_argument(
    '--text', '-x', action='store_true',
    help="print IFO as text table to stdout and exit (budget not calculated)")
group.add_argument(
    '--diff', '-d', metavar='IFO',
    help="show difference table between IFO and another IFO description (name or path) and exit (budget not calculated)")
parser.add_argument(
    '--no-plot', '-np', action='store_false', dest='plot',
    help="suppress plotting")
parser.add_argument(
    'IFO',
    help="IFO name or path")


def freq_from_spec(spec):
    fspec = spec.split(':')
    if len(fspec) == 2:
        fspec = fspec[0], FREQ.split(':')[1], fspec[1]
    return np.logspace(
        np.log10(float(fspec[0])),
        np.log10(float(fspec[2])),
        int(fspec[1]),
    )


def main():
    signal.signal(signal.SIGINT, signal.SIG_DFL)

    args = parser.parse_args()

    ##########
    # initial arg processing

    if os.path.splitext(os.path.basename(args.IFO))[1] in DATA_SAVE_FORMATS:
        if args.freq:
            parser.exit(2, "Frequency specification not allowed when loading traces from file.\n")
        if args.ifo:
            parser.exit(2, "IFO parameter specification not allowed when loading traces from file.\n")
        from .io import load_hdf5
        budget = None
        freq, traces, attrs = load_hdf5(args.IFO)
        ifo = attrs.get('ifo')
        # FIXME: deprecate 'IFO'
        ifo = attrs.get('IFO', ifo)
        plot_style = attrs

    else:
        budget = load_budget(args.IFO)
        ifo = budget.ifo
        if args.freq:
            try:
                freq = freq_from_spec(args.freq)
            except IndexError:
                parser.exit(2, "Improper frequency specification: {}\n".format(args.freq))
        else:
            freq = getattr(budget, 'freq', freq_from_spec(FREQ))
        plot_style = getattr(budget, 'plot_style', {})
        traces = None

    if args.ifo:
        for paramval in args.ifo:
            param, val = paramval.split('=', 1)
            ifo[param] = float(val)

    if args.yaml:
        if not ifo:
            parser.exit(2, "IFO structure not provided.\n")
        print(ifo.to_yaml(), end='')
        return
    if args.text:
        if not ifo:
            parser.exit(2, "IFO structure not provided.\n")
        print(ifo.to_txt(), end='')
        return
    if args.diff:
        if not ifo:
            parser.exit(2, "IFO structure not provided.\n")
        dbudget = load_budget(args.diff)
        diffs = ifo.diff(dbudget.ifo)
        if diffs:
            w = max([len(d[0]) for d in diffs])
            fmt = '{{:{}}} {{:>20}} {{:>20}}'.format(w)
            print(fmt.format('', args.IFO, args.diff))
            print(fmt.format('', '-----', '-----'))
            for p in diffs:
                k = str(p[0])
                v = repr(p[1])
                ov = repr(p[2])
                print(fmt.format(k, v, ov))
        return

    out_data_files = set()
    out_plot_files = set()
    if args.save:
        args.plot = False
        for path in args.save:
            if os.path.splitext(path)[1] in DATA_SAVE_FORMATS:
                out_data_files.add(path)
        out_plot_files = set(args.save) - out_data_files

    if args.plot or out_plot_files:
        if out_plot_files:
            # FIXME: this silliness seems to be the only way to have
            # matplotlib usable on systems without a display.  There must
            # be a better way.  'AGG' is a backend that works without
            # displays.  but it has to be set before any other matplotlib
            # stuff is imported.  and we *don't* want it set if we do want
            # to show an interactive plot.  there doesn't seem a way to
            # set this opportunistically.
            import matplotlib
            matplotlib.use('AGG')
        try:
            from matplotlib import pyplot as plt
        except RuntimeError:
            logger.warning("no display, plotting disabled.")
            args.plot = False

    try:
        import inspiral_range
        logger_ir = logging.getLogger('inspiral_range')
        logger_ir.setLevel(logger.getEffectiveLevel())
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter('%(name)s: %(message)s'))
        logger_ir.addHandler(handler)

    except ModuleNotFoundError:
        logger.warning("WARNING: inspiral_range package not available, figure of merit will not be calculated.")
        args.fom = None
    if args.fom:
        try:
            range_func, range_func_args = args.fom.split(':')
        except ValueError:
            range_func = args.fom
            range_func_args = ''
        range_params = {}
        for param in range_func_args.split(','):
            if not param:
                continue
            p, v = param.split('=')
            if not v:
                raise ValueError('missing parameter value "{}"'.format(p))
            try:
                v = float(v)
            except ValueError:
                pass
            range_params[p] = v

    ##########
    # main calculations

    if not traces:
        logger.info("calculating budget...")
        traces = budget.run(freq=freq)

    if args.title:
        plot_style['title'] = args.title
    elif 'title' in plot_style:
        pass
    elif budget:
        plot_style['title'] = "GWINC Noise Budget: {}".format(budget.name)
    else:
        plot_style['title'] = "GWINC Noise Budget: {}".format(args.IFO)

    if args.fom:
        logger.info("calculating inspiral {}...".format(range_func))
        H = inspiral_range.CBCWaveform(freq, **range_params)
        logger.debug("waveform params: {}".format(H.params))
        fom = eval('inspiral_range.{}'.format(range_func))(freq, traces['Total'][0], H=H)
        logger.info("{}({}) = {:.2f} Mpc".format(range_func, range_func_args, fom))
        fom_title = 'inspiral {func} {m1}/{m2} $\mathrm{{M}}_\odot$: {fom:.0f} Mpc'.format(
            func=range_func,
            m1=H.params['m1'],
            m2=H.params['m2'],
            fom=fom,
            )
        plot_style['title'] += '\n{}'.format(fom_title)

    ##########
    # interactive

    # interactive shell plotting
    if args.interactive:
        banner = """GWINC interactive shell

The 'ifo' Struct and 'traces' data objects are available for
inspection.  Use the 'whos' command to view the workspace.
"""
        if not args.plot:
            banner += """
You may plot the budget using the 'plot_noise()' function:

In [.]: plot_noise(freq, traces, **plot_style)
"""
        banner += """
You may interact with the plot using the 'plt' functions, e.g.:

In [.]: plt.title("foo")
In [.]: plt.savefig("foo.pdf")
"""
        from IPython.terminal.embed import InteractiveShellEmbed
        ipshell = InteractiveShellEmbed(
            banner1=banner,
            user_ns={
                'freq': freq,
                'traces': traces,
                'ifo': ifo,
                'plot_style': plot_style,
                'plot_noise': plot_noise,
            },
        )
        ipshell.enable_pylab()
        if args.plot:
            ipshell.ex("fig = plot_noise(freq, traces, **plot_style)")
            ipshell.ex("plt.title(plot_style['title'])")
        ipshell()

    ##########
    # output

    # save noise traces to HDF5 file
    if out_data_files:
        from .io import save_hdf5
        attrs = dict(plot_style)
        for path in out_data_files:
            logger.info("saving budget traces: {}".format(path))
            save_hdf5(
                path=path,
                freq=freq,
                traces=traces,
                ifo=ifo,
                **attrs,
            )

    # standard plotting
    if args.plot or out_plot_files:
        logger.debug("plotting noises...")
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        plot_noise(
            freq,
            traces,
            ax=ax,
            **plot_style
        )
        fig.tight_layout()
        if out_plot_files:
            for path in out_plot_files:
                logger.info("saving budget plot: {}".format(path))
                fig.savefig(path)
        else:
            plt.show()


if __name__ == '__main__':
    main()
