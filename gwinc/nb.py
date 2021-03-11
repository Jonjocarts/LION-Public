import operator
import itertools
import collections
import numpy as np
import scipy.interpolate
from functools import reduce

from . import logger


def quadsum(data):
    """Calculate quadrature sum of list of data arrays.

    Provided data are assumed to be power-referred, so this is a
    simple point-by-point sum.

    NaNs in sum elements do not contribute to sum.

    """
    return np.nansum(data, 0)



class BudgetItem:
    """GWINC BudgetItem class

    """
    def load(self):
        """Overload method for initial loading of static data.

        """
        return None

    def update(self, **kwargs):
        """Overload method for updating data.

        By default any keyword arguments provided are written directly
        as attribute variables (as with __init__).

        """
        for key, val in kwargs.items():
            setattr(self, key, val)

    def calc(self):
        """Overload method for final PSD calculation.

        Should return an array of power-referenced values evaluated at
        all evaluation frequencies (self.freq).

        """
        return None

    ##########

    def __init__(self, freq=None, **kwargs):
        """Initialize budget item.

        The primary argument should be the evaluation frequency array.
        If not provided, then a pre-defined `freq` attribute of the
        BudgetItem class should exist.

        Additional keyword arguments are written as attribute
        variables to the initialized object.

        """
        if freq is not None:
            assert isinstance(freq, np.ndarray)
            self.freq = freq
        for key, val in kwargs.items():
            setattr(self, key, val)

    @property
    def name(self):
        """"Name of this BudgetItem class."""
        return self.__class__.__name__

    def __str__(self):
        # FIXME: provide info on internal state (load/update/calc/etc.)
        return '<{} {}>'.format(
            ', '.join([c.__name__ for c in self.__class__.__bases__]),
            self.name,
        )

    def interpolate(self, freq, data):
        """Interpolate data to the evaluation frequencies.

        """
        func = scipy.interpolate.interp1d(
            freq, data,
            kind='nearest',
            copy=False,
            assume_sorted=True,
            bounds_error=False,
            fill_value=np.nan,
        )
        return func(self.freq)


class Calibration(BudgetItem):
    """GWINC Calibration class

    BudgetItem that represents a calibration transfer function for a
    Noise.  The calc() method should return a transfer function
    amplitude array evaluated at the evaluation frequencies supplied
    at initialization and available in the `freq` array attribute
    (self.freq).

    """
    def __call__(self, data):
        """Calibrate input data.

        Returns calibrated version of input data array,
        e.g. point-by-point product of data and calibration arrays.

        """
        cal = self.calc()
        assert data.shape == cal.shape, \
            "data shape does not match calibration ({} != {})".format(data.shape, cal.shape)
        return data * cal


class Noise(BudgetItem):
    """GWINC Noise class

    BudgetItem that represents a PSD noise calculation.  The calc()
    method should return the noise PSD spectrum array evaluated at the
    evaluation frequencies supplied at initialization and available in
    the `freq` array attribute (self.freq).

    """

    style = {}
    """Trace plot style dictionary"""

    def calc_trace(self, calibrations=None, calc=True):
        """Returns noise (PSD, style) tuple.

        If `calibrations` is not None it is assumed to be a list of
        len(self.freq) arrays that will be multiplied to the output
        PSD.

        If calc=False, the noise will not be calculated and the PSD
        will be None.  This is useful for just getting the style.

        """
        if calc:
            data = self.calc()
            if calibrations:
                data *= reduce(
                    operator.mul,
                    calibrations,
                )
        else:
            data = None
        return data, self.style

    def run(self, **kwargs):
        """Convenience method to load, update, and return calc traces.

        Equivalent of load(), update(), calc_traces() run in sequence.
        Keyword arguments are passed to update().

        """
        self.load()
        self.update(**kwargs)
        return self.calc_trace()


class Budget(Noise):
    """GWINC Budget class

    This is a Noise that represents a budget of multiple sub noises.

    The `noises` attribute of this class should list constituent Noise
    classes.  Each element can be either a single Noise class, or a
    tuple of (Noise, Calibration) classes, e.g.:

    noises = [
        Thermal,
        (Shot, Sensing),
    ]

    When this object is initialized, all sub noises and calibrations
    are initialized.  Pre-defined load() and update() methods call the
    load() and update() methods of all sub noises and calibrations.
    When calc() is called, the PSD is calculated for all sub noises,
    the relevant calibration is evaluated and applied, and the
    quadrature sum of all calibrated consituent noises is returned.

    Additionally, a `calibrations` attribute may define a list of
    common calibrations to apply to all noises, e.g.:

    calibrations = [
        Strain,
    ]

    Finally, a `references` attribute may be defined, similar to the
    `noises` attribute described above except that the specified
    noises do not contribute to the overall budget total, e.g.:

    references = [
        strain_data_20200120,
    ]

    NOTE: if an `ifo` attribute is defined it is always passed as an
    initialization argument to sub noises.

    """

    noises = []
    """List of constituent noise classes, or (noise, cal) tuples"""

    calibrations = []
    """List of calibrations to be applied to all budget noises (not references)"""

    references = []
    """List of reference noise classes, or (ref, cal) tuples"""

    def __init__(self, freq=None, noises=None, **kwargs):
        """Initialize Budget object.

        See BudgetItem for base initialization arguments.

        If a `noises` keyword argument is provided it should be an
        iterable of noise names (constituent or reference) which will
        be used to filter the noises initialized in this budget.

        """
        super().__init__(freq, **kwargs)
        # store kwargs for later use
        self.kwargs = kwargs
        # record the frequency array as a kwarg if it's definied as a
        # class attribute
        if freq is not None:
            self.kwargs['freq'] = freq
        else:
            self.kwargs['freq'] = getattr(self, 'freq', None)
        # FIXME: special casing the ifo kwarg here, in case it's
        # defined as a class attribute rather than passed at
        # initialization.  we do this because we're not defining a
        # standard way to extract IFO variables that get passed around
        # in a reasonable way.  how can we clarify this?
        if 'ifo' not in kwargs and getattr(self, 'ifo', None):
            self.kwargs['ifo'] = getattr(self, 'ifo', None)
        # all noise objects keyed by name
        self._noise_objs = collections.OrderedDict()
        # all cal objects keyed by name
        self._cal_objs = {}
        # set of calibration names to apply to noise
        self._noise_cals = collections.defaultdict(set)
        # set of all constituent budget noise names
        self._budget_noises = set()
        # initialize all noise objects
        for nc in self.noises:
            name = self.__init_noise(nc, noises)
            if name:
                self._budget_noises.add(name)
        # initialize common calibrations and add to all budget noises
        for cal in self.calibrations:
            self.__add_calibration(cal, self._budget_noises)
        # initialize references, without common calibrations
        for nc in self.references:
            self.__init_noise(nc, noises)
        # error if requested noise is not present
        if noises:
            sset = set(noises)
            nset = set([name for name in self._noise_objs.keys()])
            if not sset <= nset:
                raise AttributeError("unknown noise terms: {}".format(' '.join(sset-nset)))

    def __init_noise(self, nc, noise_filt):
        cal = None
        if isinstance(nc, (list, tuple)):
            noise = nc[0]
            cals = nc[1:]
        else:
            noise = nc
            cals = []
        noise_obj = noise(
            **self.kwargs
        )
        name = noise_obj.name
        if noise_filt and name not in noise_filt:
            return
        logger.debug("init {}".format(noise_obj))
        self._noise_objs[name] = noise_obj
        for cal in cals:
            self.__add_calibration(cal, [name])
        return name

    def __add_calibration(self, cal, noises):
        cal_obj = cal(
            **self.kwargs
        )
        name = cal_obj.name
        if name not in self._cal_objs:
            logger.debug("init {}".format(cal_obj))
            self._cal_objs[name] = cal_obj
        # register noises for this calibration
        for noise in noises:
            self._noise_cals[noise].add(name)
        return name

    def __getitem__(self, name):
        try:
            return self._noise_objs[name]
        except KeyError:
            try:
                return self._cal_objs[name]
            except KeyError:
                raise KeyError("unknown noise or cal name '{}".format(name))

    def keys(self):
        """Iterate over budget noise names."""
        return iter(self._noise_objs.keys())

    def values(self):
        """Iterate over budget noise objects."""
        return iter(self._noise_objs.values())

    def items(self):
        """Iterate over budget noise (name, object) tuples."""
        return iter(self._noise_objs.items())

    def __iter__(self):
        return iter(self.keys())

    def walk(self):
        """Walk recursively through every BudgetItem in the budget.

        This includes Noise, Calibration and Budget objects, as well
        as any decendents of Budget objects.

        For each leaf item yields a tuple of all ancestor objects,
        e.g.:

          (self)
          (self, BudgetItem)
          (self, ChildBudget1)
          (self, ChildBudget1, BudgetItem)
          ...

        """
        yield (self,)
        for item in itertools.chain(
                self._cal_objs.values(),
                self._noise_objs.values()):
            if isinstance(item, Budget):
                for i in item.walk():
                    yield (self,) + i
            else:
                yield (self, item)

    def load(self):
        """Load all noise and cal objects."""
        for name, item in itertools.chain(
                self._cal_objs.items(),
                self._noise_objs.items()):
            logger.debug("load {}".format(item))
            item.load()

    def update(self, **kwargs):
        """Update all noise and cal objects with supplied kwargs."""
        for name, item in itertools.chain(
                self._cal_objs.items(),
                self._noise_objs.items()):
            logger.debug("update {}".format(item))
            item.update(**kwargs)

    def calc_noise(self, name):
        """Return calibrated individual noise term.

        The noise PSD and calibration transfer functions are
        calculated, and the calibrated noise array is returned.

        """
        noise = self._noise_objs[name]
        cals = [self._cal_objs[cal].calc() for cal in self._noise_cals[name]]
        data = noise.calc_trace(cals)
        if isinstance(data, dict):
            return data['Total'][0]
        else:
            return data[0]

    def calc(self):
        """Calculate sum of all noises.

        """
        data = [self.calc_noise(name) for name in self._noise_objs.keys() if name in self._budget_noises]
        return quadsum(data)

    def calc_trace(self, calibrations=None, calc=True):
        """Returns a dictionary of noises traces, keyed by noise names.

        Values are (data, style) trace tuples (see Noise.calc_trace).
        The key of the budget sum total is 'Total'.  The values of sub
        budgets are themselves dictionaries returned from
        calc_trace() of the sub budget.

        If `calibrations` is not None it is assumed to be a list of
        len(self.freq) array that will be multiplied to the output PSD
        of the budget and all sub noises.

        If calc=False, the noise will not be calculated and the PSD
        will be None.  This is useful for just getting style the
        style.

        """
        # start by creating an empty OrderedDict used for outputing trace data
        # or style info with the following order:
        #   references
        #   total
        #   constituents
        d = collections.OrderedDict()
        # allocate references
        for name, noise in self._noise_objs.items():
            if name in self._budget_noises:
                continue
            d[name] = noise.calc_trace(calc=False)
        # allocate total
        if self._budget_noises:
            d['Total'] = None, self.style
        # allocate constituent
        for name, noise in self._noise_objs.items():
            if name not in self._budget_noises:
                continue
            d[name] = noise.calc_trace(calc=False)
        # if we're not calc'ing, just return the dict now
        if not calc:
            return d

        # calc all calibrations
        c = {}
        for name, cal in self._cal_objs.items():
            logger.debug("calc {}".format(name))
            c[name] = cal.calc()
        # calc all noises
        for name, noise in self._noise_objs.items():
            cals = [c[cal] for cal in self._noise_cals[name]]
            if calibrations:
                cals += calibrations
            logger.debug("calc {}".format(name))
            d[name] = noise.calc_trace(
                calibrations=cals,
            )
        # calc budget total
        constituent_data = []
        for name in self._budget_noises:
            if isinstance(d[name], dict):
                data = d[name]['Total'][0]
            else:
                data = d[name][0]
            constituent_data.append(data)
        d['Total'] = quadsum(constituent_data), self.style
        return d
