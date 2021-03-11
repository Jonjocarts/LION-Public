import os


IFOS = [
    'aLIGO',
    'Aplus',
    'Voyager',
    'CE1',
    'CE2',
    'LION',
]


PLOT_STYLE = dict(
    ylabel=u"Strain [1/\u221AHz]",
)


def lpath(file0, file1):
    """Return path of file1 when expressed relative to file0.

    For instance, if file0 is "/path/to/file0" and file1 is
    "../for/file1" then what is returned is "/path/for/file1".

    This is useful for resolving paths within packages with e.g.:

      rpath = lpath(__file__, '../resource.yaml')

    """
    return os.path.abspath(os.path.join(os.path.dirname(file0), file1))
