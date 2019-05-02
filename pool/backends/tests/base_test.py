"""Test suite for analysis backends."""
from builtins import object
from six import with_metaclass

from abc import ABCMeta, abstractmethod
from copy import deepcopy
from numpy.testing import \
    run_module_suite, assert_array_equal, assert_equal, SkipTest
import numpy as np
import pandas as pd
import uuid


class BaseTests(with_metaclass(ABCMeta, object)):

    @abstractmethod
    def setup(self):
        raise NotImplementedError

    @abstractmethod
    def teardown(self):
        raise NotImplementedError

    def test_none(self):
        analysis_name = 'test_none'
        val = None
        self.db.store(
            analysis_name=analysis_name, data=val, keys=self.keys,
            updated=self.updated)
        val2, updated2, dependencies2 = \
            self.db.recall(analysis_name=analysis_name, keys=self.keys)
        assert_equal(val2, val)

    def test_float(self):
        analysis_name = 'test_float'
        val = 1.0
        self.db.store(
            analysis_name=analysis_name, data=val, keys=self.keys,
            updated=self.updated)
        val2, updated2, dependencies2 = \
            self.db.recall(analysis_name=analysis_name, keys=self.keys)
        assert_equal(val2, val)

    def test_np_float(self):
        analysis_name = 'test_np_float'
        val = np.float64(1.0)
        self.db.store(
            analysis_name=analysis_name, data=val, keys=self.keys,
            updated=self.updated)
        val2, updated2, dependencies2 = \
            self.db.recall(analysis_name=analysis_name, keys=self.keys)
        assert_equal(val2, val)

    def test_int(self):
        analysis_name = 'test_int'
        val = 1
        self.db.store(
            analysis_name=analysis_name, data=val, keys=self.keys,
            updated=self.updated)
        val2, updated2, dependencies2 = \
            self.db.recall(analysis_name=analysis_name, keys=self.keys)
        assert_equal(val2, val)

    def test_np_int(self):
        analysis_name = 'test_np_int'
        val = np.int64(1)
        self.db.store(
            analysis_name=analysis_name, data=val, keys=self.keys,
            updated=self.updated)
        val2, updated2, dependencies2 = \
            self.db.recall(analysis_name=analysis_name, keys=self.keys)
        assert_equal(val2, val)

    def test_str(self):
        analysis_name = 'test_str'
        val = 'test string'
        self.db.store(
            analysis_name=analysis_name, data=val, keys=self.keys,
            updated=self.updated)
        val2, updated2, dependencies2 = \
            self.db.recall(analysis_name=analysis_name, keys=self.keys)
        assert_equal(val2, val)

    def test_bool(self):
        analysis_name = 'test_bool'
        val = True
        self.db.store(
            analysis_name=analysis_name, data=val, keys=self.keys,
            updated=self.updated)
        val2, updated2, dependencies2 = \
            self.db.recall(analysis_name=analysis_name, keys=self.keys)
        assert_equal(val2, val)

    def test_np_bool(self):
        analysis_name = 'test_np_bool'
        val = np.bool_(True)
        self.db.store(
            analysis_name=analysis_name, data=val, keys=self.keys,
            updated=self.updated)
        val2, updated2, dependencies2 = \
            self.db.recall(analysis_name=analysis_name, keys=self.keys)
        assert_equal(val2, val)

    def test_list(self):
        analysis_name = 'test_list'
        val = [1, 2., 'foo', np.int64(1), np.float64(2.)]
        self.db.store(
            analysis_name=analysis_name, data=val, keys=self.keys,
            updated=self.updated)
        val2, updated2, dependencies2 = \
            self.db.recall(analysis_name=analysis_name, keys=self.keys)
        assert_equal(val2, val)

    def test_dict(self):
        analysis_name = 'test_dict'
        val = {'first': 1, 'second': 2., 'third': 'foo', 'fourth': np.int64(1),
               'fifth': np.float64(2.)}
        self.db.store(
            analysis_name=analysis_name, data=val, keys=self.keys,
            updated=self.updated)
        val2, updated2, dependencies2 = \
            self.db.recall(analysis_name=analysis_name, keys=self.keys)
        assert_equal(val2, val)

    def test_array(self):
        analysis_name = 'test_array'
        val = np.array([1, 2., 'foo', np.int64(1), np.float64(2.)])
        self.db.store(
            analysis_name=analysis_name, data=val, keys=self.keys,
            updated=self.updated)
        val2, updated2, dependencies2 = \
            self.db.recall(analysis_name=analysis_name, keys=self.keys)
        assert_equal(val2, val)

    def test_nan(self):
        analysis_name = 'test_nan'
        val = np.nan
        self.db.store(
            analysis_name=analysis_name, data=val, keys=self.keys,
            updated=self.updated)
        val2, updated2, dependencies2 = \
            self.db.recall(analysis_name=analysis_name, keys=self.keys)
        assert_equal(val2, val)

    def test_nan_list(self):
        analysis_name = 'test_nan_list'
        val = [np.nan, 1, 2.]
        self.db.store(
            analysis_name=analysis_name, data=val, keys=self.keys,
            updated=self.updated)
        val2, updated2, dependencies2 = \
            self.db.recall(analysis_name=analysis_name, keys=self.keys)
        assert_equal(val2, val)

    def test_nan_dict(self):
        analysis_name = 'test_nan_dict'
        val = {'first': np.nan, 'second': 1, 'third': 2.}
        self.db.store(
            analysis_name=analysis_name, data=val, keys=self.keys,
            updated=self.updated)
        val2, updated2, dependencies2 = \
            self.db.recall(analysis_name=analysis_name, keys=self.keys)
        assert_equal(val2, val)

    def test_nan_array(self):
        analysis_name = 'test_nan_array'
        val = np.array([np.nan, 1, 2.])
        self.db.store(
            analysis_name=analysis_name, data=val, keys=self.keys,
            updated=self.updated)
        val2, updated2, dependencies2 = \
            self.db.recall(analysis_name=analysis_name, keys=self.keys)
        assert_equal(val2, val)

    def test_array_list(self):
        analysis_name = 'test_array_list'
        val = [1, 'test', np.arange(3)]
        self.db.store(
            analysis_name=analysis_name, data=val, keys=self.keys,
            updated=self.updated)
        val2, updated2, dependencies2 = \
            self.db.recall(analysis_name=analysis_name, keys=self.keys)
        assert_equal(val2, val)

    def test_array_dict(self):
        analysis_name = 'test_array_dict'
        val = {'first': 1, 'second': 'test', 'third': np.arange(3)}
        self.db.store(
            analysis_name=analysis_name, data=val, keys=self.keys,
            updated=self.updated)
        val2, updated2, dependencies2 = \
            self.db.recall(analysis_name=analysis_name, keys=self.keys)
        assert_equal(val2, val)

    def test_df(self):
        analysis_name = 'test_df'
        val = (pd
               .DataFrame({'value': [1, 2], 'mouse': ['TM001', 'TM001']})
               .set_index('mouse')
               )
        self.db.store(
            analysis_name=analysis_name, data=val, keys=self.keys,
            updated=self.updated)
        val2, updated2, dependencies2 = \
            self.db.recall(analysis_name=analysis_name, keys=self.keys)
        assert_array_equal(val2.index, val.index)
        assert_array_equal(val2.values, val.values)

    def test_dependencies(self):
        analysis_name = 'test_depends'
        val = 2.
        depends_on = {'test_float': self.updated}
        self.db.store(
            analysis_name=analysis_name, data=val, keys=self.keys,
            updated=self.updated, depends_on=depends_on)
        val2, updated2, dependcies2 = \
            self.db.recall(analysis_name=analysis_name, keys=self.keys)
        assert_equal(val2, val)
        assert_equal(dependcies2, depends_on)

    def test_missing_key(self):
        analysis_name = 'test_missing_key'
        keys = deepcopy(self.keys)
        keys['mouse'] = str(uuid.uuid1())
        val2, updated2, dependencies2 = \
            self.db.recall(analysis_name=analysis_name, keys=keys)
        assert_equal(val2, None)
        assert_equal(updated2, None)
        assert_equal(dependencies2, None)

    def test_parameters(self):
        analysis_name = 'test_parameters'
        keys = deepcopy(self.keys)
        keys['foo'] = 'bar'
        keys['dogs'] = 7
        val = 42
        self.db.store(
            analysis_name=analysis_name, data=val, keys=keys,
            updated=self.updated)
        val2, updated2, dependcies2 = \
            self.db.recall(analysis_name=analysis_name, keys=keys)
        assert_equal(val2, val)

    def test_delete_mouse(self):
        try:
            self.db.delete_mouse('mouse')
        except NotImplementedError:
            raise SkipTest
        except:
            raise
        analysis_name = 'test_delete_mouse'
        keys = deepcopy(self.keys)
        keys['mouse'] = 'TM007'
        val = 42
        self.db.store(
            analysis_name=analysis_name, data=val, keys=keys,
            updated=self.updated)
        val1, updated1, dependencies1 = \
            self.db.recall(analysis_name=analysis_name, keys=keys)
        assert_equal(val1, val)
        assert_equal(updated1, self.updated)
        assert_equal(dependencies1, {})
        self.db.delete_mouse(keys['mouse'], no_action=False)
        val2, updated2, dependencies2 = \
            self.db.recall(analysis_name=analysis_name, keys=keys)
        assert_equal(val2, None)
        assert_equal(updated2, None)
        assert_equal(dependencies2, None)

    def test_delete_analysis(self):
        try:
            self.db.delete_analysis('analysis')
        except NotImplementedError:
            raise SkipTest
        except:
            raise
        analysis_name = 'test_delete_analysis'
        keys = deepcopy(self.keys)
        keys['mouse'] = 'TM007'
        val = 42
        self.db.store(
            analysis_name=analysis_name, data=val, keys=keys,
            updated=self.updated)
        val1, updated1, dependencies1 = \
            self.db.recall(analysis_name=analysis_name, keys=keys)
        assert_equal(val1, val)
        assert_equal(updated1, self.updated)
        assert_equal(dependencies1, {})
        self.db.delete_analysis(analysis_name, no_action=False)
        val2, updated2, dependencies2 = \
            self.db.recall(analysis_name=analysis_name, keys=keys)
        assert_equal(val2, None)
        assert_equal(updated2, None)
        assert_equal(dependencies2, None)


if __name__ == "__main__":
    run_module_suite()
