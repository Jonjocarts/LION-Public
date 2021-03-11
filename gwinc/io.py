import h5py
import yaml
import datetime

from . import logger
from . import Struct


SCHEMA = 'GWINC noise budget'
SCHEMA_VERSION = 1


def _write_trace_recursive(grp, traces):
    for name, trace in traces.items():
        if isinstance(trace, dict):
            tgrp = grp.create_group(name)
            _write_trace_recursive(tgrp, trace)
        else:
            data, style = trace
            dset = grp.create_dataset(name, data=data)
            for key, val in style.items():
                dset.attrs[key] = val


def save_hdf5(path, freq, traces, **kwargs):
    """Save GWINC budget data to an HDF5 file.

    The `freq` argument should be the frequency array, and `traces`
    should be the traces (recursive) dictionary.  Keyword arguments
    are stored in the HDF5 top level 'attrs' key-value store.  If an
    'ifo' keyword arg is supplied, it is assumed to be a Struct and
    will be serialized to YAML for storage.

    See HDF5_SCHEMA.

    """
    with h5py.File(path, 'w') as f:
        f.attrs['SCHEMA'] = SCHEMA
        f.attrs['SCHEMA_VERSION'] = SCHEMA_VERSION
        # FIXME: add budget code hash or something
        f.attrs['date'] = datetime.datetime.now().isoformat()
        for key, val in kwargs.items():
            if key == 'ifo':
                f.attrs['ifo'] = val.to_yaml()
            else:
                f.attrs[key] = val
        f.create_dataset('Freq', data=freq)
        tgrp = f.create_group('traces')
        _write_trace_recursive(tgrp, traces)


def _read_trace_recursive(element):
    trace = {}
    for name, item in element.items():
        if isinstance(item, h5py.Group):
            trace[name] = _read_trace_recursive(item)
        else:
            trace[name] = item[:], dict(item.attrs.items())
    return trace


def load_hdf5(path):
    """Load GWINC budget data from an HDF5 file.

    Returns (freq, traces, attrs).  An 'ifo' attr will be
    de-serialized from YAML into a Struct object.

    See HDF5_SCHEMA.

    """
    with h5py.File(path, 'r') as f:
        # FIXME: check SCHEMA name/version
        freq = f['Freq'][:]
        traces = _read_trace_recursive(f['/traces'])
        attrs = dict(f.attrs)
        if 'ifo' in attrs:
            try:
                attrs['ifo'] = Struct.from_yaml(attrs['ifo'])
            except yaml.constructor.ConstructorError:
                logger.warning("HDF5 load warning: Could not de-serialize 'ifo' YAML attribute.")
        return freq, traces, attrs
