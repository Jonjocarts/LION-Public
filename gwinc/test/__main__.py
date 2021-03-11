import os
import sys
import shutil
import signal
import logging
import tempfile
import argparse
import subprocess
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict
from collections.abc import Mapping
from PyPDF2 import PdfFileReader, PdfFileWriter

from .. import IFOS, load_budget
from ..io import load_hdf5

logging.basicConfig(
    format='%(message)s',
    level=os.getenv('LOG_LEVEL', 'INFO').upper())

try:
    import inspiral_range
except ImportError:
    logging.warning("inspiral_range package not found, range will not be calculated")
    inspiral_range = None


TOLERANCE = 1e-6
CACHE_LIMIT = 5


def test_path(*args):
    """Return path to package file."""
    return os.path.join(os.path.dirname(__file__), *args)


def git_find_upstream_name():
    try:
        remotes = subprocess.run(
            ['git', 'remote', '-v'],
            capture_output=True, universal_newlines=True,
            check=True,
        ).stdout
    except subprocess.CalledProcessError as e:
        logging.error(e.stderr.split('\n')[0])
    for remote in remotes.strip().split('\n'):
        name, url, fp = remote.split()
        if 'gwinc/pygwinc.git' in url:
            return name


def git_rev_resolve_hash(git_rev):
    """Resolve a git revision into its hash string."""
    try:
        return subprocess.run(
            ['git', 'show', '-s', '--format=format:%H', git_rev],
            capture_output=True, universal_newlines=True,
            check=True,
        ).stdout
    except subprocess.CalledProcessError as e:
        logging.error(e.stderr.split('\n')[0])


def prune_cache_dir():
    """Prune all but the N most recently accessed caches.

    """
    cache_dir = test_path('cache')
    if not os.path.exists(cache_dir):
        return
    expired_paths = sorted(
        [os.path.join(cache_dir, path) for path in os.listdir(cache_dir)],
        key=lambda path: os.stat(path).st_atime, reverse=True,
    )[CACHE_LIMIT:]
    if not expired_paths:
        return
    for path in expired_paths:
        logging.info("pruning old cache: {}".format(path))
        shutil.rmtree(path)


def gen_cache(git_hash, path):
    """generate cache for specified git hash at the specified path

    The included shell script is used to extract the gwinc code from
    the appropriate git commit, and invoke a new python instance to
    generate the noise curves.

    """
    logging.info("creating new cache for hash {}...".format(git_hash))
    subprocess.run(
        [test_path('gen_cache.sh'), git_hash, path, sys.executable],
        check=True,
    )


def load_cache(path):
    """load a cache from the specified path

    returns a "cache" dictionary with 'git_hash' and 'ifos' keys.

    """
    logging.info("loading cache {}...".format(path))
    cache = {}
    git_hash_path = os.path.join(path, 'git_hash')
    if os.path.exists(git_hash_path):
        with open(git_hash_path) as f:
            git_hash = f.read().strip()
    else:
        git_hash = None
    logging.debug("cache hash: {}".format(git_hash))
    cache['git_hash'] = git_hash
    cache['ifos'] = {}
    for f in sorted(os.listdir(path)):
        name, ext = os.path.splitext(f)
        if ext != '.h5':
            continue
        cache['ifos'][name] = os.path.join(path, f)
    return cache


def walk_traces(traces, root=()):
    """recursively walk through traces

    yields (key_tuple, value), where key_tuple is a tuple of nested
    dict keys of the traces dict locating the given value.

    """
    for key, val in traces.items():
        r = root+(key,)
        if isinstance(val, Mapping):
            for k, v in walk_traces(val, root=r):
                yield k, v
            continue
        else:
            yield r, val


def zip_noises(tracesA, tracesB, skip):
    """zip matching noises from traces"""
    for keys, (noiseA, _) in walk_traces(tracesA):
        # keys is a tuple of nested keys for noise
        name = keys[-1]
        # skip keys we'll add at the end
        if name in ['Total', 'int73']:
            continue
        if skip and name in skip:
            logging.warning("SKIPPING TEST: '{}'".format(name))
            continue

        try:
            t = tracesB
            for key in keys:
                t = t[key]
            noiseB, style = t
        except (KeyError, TypeError):
            logging.warning("MISSING B TRACE: '{}'".format(name))
            continue

        yield name, noiseA, noiseB

    yield 'Total', tracesA['Total'][0], tracesB['Total'][0]

    if 'int73' in tracesA:
        yield 'int73', tracesA['int73'][0], tracesB['int73'][0]


def compare_traces(tracesA, tracesB, tolerance=TOLERANCE, skip=None):
    """Compare two gwinc trace dictionaries

    Noises listed in `skip` will not be compared.

    Returns a dictionary of noises that differ fractionally (on a
    point-by-point basis) by more that `tolerance` between `tracesA`
    and `tracesB`.

    """
    #name_width = max([len(n[0][-1]) for n in walk_traces(tracesA)])
    name_width = 15
    diffs = OrderedDict()
    for name, noiseA, noiseB in zip_noises(tracesA, tracesB, skip):
        logging.debug("comparing {}...".format(name))

        ampA = np.sqrt(noiseA)
        ampB = np.sqrt(noiseB)
        diff = ampB - ampA
        frac = abs(diff / ampA)

        if max(frac) < tolerance:
            continue

        logging.warning("EXCESSIVE DIFFERENCE: {:{w}} {:6.1f} ppm".format(
            name, max(frac)*1e6, w=name_width))

        diffs[name] = (noiseA, noiseB, frac)

    return diffs


def plot_diffs(freq, diffs, styleA, styleB):
    spec = (len(diffs)+1, 2)
    sharex = None
    for i, nname in enumerate(diffs):
        noiseA, noiseB, frac = diffs[nname]

        axl = plt.subplot2grid(spec, (i, 0), sharex=None)
        axl.loglog(freq, np.sqrt(noiseA), **styleA)
        axl.loglog(freq, np.sqrt(noiseB), **styleB)
        axl.grid()
        axl.legend(loc='upper right')
        axl.set_ylabel(nname)
        if i == 0:
            axl.set_title("noise value")

        if i == 0:
            sharex = axl
        axr = plt.subplot2grid(spec, (i, 1), sharex=sharex)
        axr.loglog(freq, frac)
        axr.grid()
        axr.axhline(y=max(frac), color='r', linestyle='--')
        axr.text(max(freq)+4000, max(frac), '{:.1f} ppm'.format(max(frac)*1e6),
                 horizontalalignment='left', verticalalignment='center',
                 color='red')
        if i == 0:
            axr.set_title("fractional difference")

    axl.set_xlabel("frequency [Hz]")
    axr.set_xlabel("frequency [Hz]")
    plt.subplots_adjust(top=0.8, right=0.85, wspace=0.3)

##################################################

def main():
    parser = argparse.ArgumentParser(
        description="""GWINC noise validation

This command calculates the canonical noise budgets with the current
code and compares them against those calculated with code from a
specified git revision.  You must be running from a git checkout of
the source for this to work.  The command will fail if it detects any
noise differences.  Plots or a PDF report of differences can be
generated with the '--plot' or '--report' commands respectively.

By default it will attempt to determine the git reference for upstream
master for your current configuration (usually 'origin/master' or
'upstream/master').  You may specify an arbitrary git revision with
the --git-rev command.  For example, to compare against another
remote/branch use:

$ python3 -m gwinc.test --git-rev remote/dev-branch

or if you have uncommitted changes compare against the current head
with:

$ python3 -m gwinc.test -g HEAD

See gitrevisions(7) for various ways to refer to git revisions.

A cache of traces from reference git revisions will be stored in
gwinc/test/cache/<SHA1>.  Old caches are automatically pruned.""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        '--tolerance', '-t',  type=float, default=TOLERANCE,
        help="fractional tolerance of comparison [{}]".format(TOLERANCE))
    parser.add_argument(
        '--skip', '-k', metavar='NOISE', action='append',
        help="traces to skip in comparison (multiple may be specified)")
    rgroup = parser.add_mutually_exclusive_group()
    rgroup.add_argument(
        '--git-rev', '-g', metavar='REV',
        help="specify specific git revision to compare against")
    rgroup.add_argument(
        '--head', '-gh', action='store_const', dest='git_rev', const='HEAD',
        help="shortcut for '--git-rev HEAD'")
    ogroup = parser.add_mutually_exclusive_group()
    ogroup.add_argument(
        '--plot', '-p', action='store_true',
        help="show interactive plot differences")
    ogroup.add_argument(
        '--report', '-r', metavar='REPORT.pdf',
        help="create PDF report of test results (only created if differences found)")
    parser.add_argument(
        'ifo', metavar='IFO', nargs='*',
        help="specific ifos to test (default all)")
    args = parser.parse_args()

    # get the reference hash
    if args.git_rev:
        git_rev = args.git_rev
    else:
        remote = git_find_upstream_name()
        if not remote:
            sys.exit("Could not resolve upstream remote name")
        git_rev = '{}/master'.format(remote)
    logging.info("git  rev: {}".format(git_rev))
    git_hash = git_rev_resolve_hash(git_rev)
    if not git_hash:
        sys.exit("Could not resolve reference, could not run test.")
    logging.info("git hash: {}".format(git_hash))

    # load the cache
    cache_path = test_path('cache', git_hash)
    if not os.path.exists(cache_path):
        prune_cache_dir()
        gen_cache(git_hash, cache_path)
    cache = load_cache(cache_path)

    if args.report:
        base, ext = os.path.splitext(args.report)
        if ext != '.pdf':
            parser.error("Test reports only support PDF format.")
        outdir = tempfile.TemporaryDirectory()

    if args.ifo:
        ifos = args.ifo
    else:
        ifos = IFOS

    style_ref = dict(label='reference', linestyle='-')
    style_cur = dict(label='current', linestyle='--')

    fail = False

    # compare
    for name in ifos:
        logging.info("{} tests...".format(name))

        try:
            path = cache['ifos'][name]
        except KeyError:
            logging.warning("IFO {} not found in cache")
            fail |= True
            continue

        freq, traces_ref, attrs = load_hdf5(path)

        budget = load_budget(name, freq)
        traces_cur = budget.run()

        if inspiral_range:
            total_ref = traces_ref['Total'][0]
            total_cur = traces_cur['Total'][0]
            range_func = inspiral_range.range
            H = inspiral_range.waveform.CBCWaveform(freq)
            fom_ref = range_func(freq, total_ref, H=H)
            traces_ref['int73'] = inspiral_range.int73(freq, total_ref)[1], None
            fom_cur = range_func(freq, total_cur, H=H)
            traces_cur['int73'] = inspiral_range.int73(freq, total_cur)[1], None
            fom_summary = """
inspiral {func} {m1}/{m2} Msol:
{label_ref}: {fom_ref:.2f} Mpc
{label_cur}: {fom_cur:.2f} Mpc
""".format(
                func=range_func.__name__,
                m1=H.params['m1'],
                m2=H.params['m2'],
                label_ref=style_ref['label'],
                fom_ref=fom_ref,
                label_cur=style_cur['label'],
                fom_cur=fom_cur,
            )
        else:
            fom_summary = ''

        diffs = compare_traces(traces_ref, traces_cur, args.tolerance, args.skip)

        if not diffs:
            logging.info("{} tests pass.".format(name))
            continue

        logging.warning("{} tests FAIL".format(name))
        fail |= True
        if args.plot or args.report:
            plot_diffs(freq, diffs, style_ref, style_cur)
            plt.suptitle('''{} {}/{} noise comparison
(noises that differ by more than {} ppm)
reference git hash: {}
{}'''.format(name, style_ref['label'], style_cur['label'],
             args.tolerance*1e6, cache['git_hash'], fom_summary))
            if args.report:
                pwidth = 10
                pheight = (len(diffs) * 5) + 2
                plt.gcf().set_size_inches(pwidth, pheight)
                plt.savefig(os.path.join(outdir.name, name+'.pdf'))
            else:
                plt.show()

    if not fail:
        logging.info("all tests pass.")
        return 0

    if args.report:
        logging.info("generating report {}...".format(args.report))
        pdf_writer = PdfFileWriter()
        for name in ifos:
            path = os.path.join(outdir.name, '{}.pdf'.format(name))
            if not os.path.exists(path):
                continue
            pdf_reader = PdfFileReader(path)
            for page in range(pdf_reader.getNumPages()):
                pdf_writer.addPage(pdf_reader.getPage(page))
        with open(args.report, 'wb') as f:
            pdf_writer.write(f)
        outdir.cleanup()

    logging.info("TESTS FAILED.")
    return 1

##################################################

if __name__ == '__main__':
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    sys.exit(main())
