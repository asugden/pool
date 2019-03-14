import numpy as np
import pandas as pd
import shutil
from tempfile import mkdtemp
import time
import uuid

import pool

def test_store_float(db, verbose=True):
    n = 1000
    repeats = 3
    data = 1.
    updated = 111111
    depends_on = {'other_test': 222222}
    times = 0
    for r in xrange(repeats):
        all_keys = [{'date': 180101,
                     'mouse': 'TM001',
                     'run': 1,
                     'foo': str(uuid.uuid4())} for x in xrange(n)]
        start_s = time.time()
        for keys in all_keys:
            db.store(
                analysis_name='test_store_float', data=data, keys=keys,
                updated=updated, depends_on=depends_on)
        times += (time.time() - start_s) / n
    if verbose:
        print(db)
        print(" {} - {} repeats of {}".format(times / repeats, repeats, n))
    return times / repeats, n, repeats


def test_replace_float(db, verbose=True):
    n = 1000
    repeats = 3
    data = 1.
    keys = {'date': 180101,
            'mouse': 'TM001',
            'run': 1,
            'foo': 'bar'}
    updated = 111111
    depends_on = {'other_test': 222222}
    db.store(
        analysis_name='test_replace_float', data=data, keys=keys,
        updated=updated, depends_on=depends_on)
    times = 0
    for r in xrange(repeats):
        start_s = time.time()
        for _ in xrange(n):
            db.store(
                analysis_name='test_replace_float', data=data, keys=keys,
                updated=updated, depends_on=depends_on)
        times += (time.time() - start_s) / n
    if verbose:
        print(db)
        print(" {} - {} repeats of {}".format(times / repeats, repeats, n))
    return times / repeats, n, repeats

def test_recall_float(db, verbose=True):
    n = 1000
    repeats = 3
    data = 1.
    keys = {'date': 180101,
            'mouse': 'TM001',
            'run': 1,
            'foo': 'bar'}
    updated = 111111
    depends_on = {'other_test': 222222}
    db.store(
        analysis_name='test_recall_float', data=data, keys=keys,
        updated=updated, depends_on=depends_on)
    times = 0
    for r in xrange(repeats):
        start_s = time.time()
        for _ in xrange(n):
            db.recall(analysis_name='test_recall_float', keys=keys)
        times += (time.time() - start_s) / n
    if verbose:
        print(db)
        print(" {} - {} repeats of {}".format(times / repeats, repeats, n))
    return times / repeats, n, repeats

def test_store_array(db, verbose=True):
    n = 1000
    repeats = 3
    data = np.arange(200)
    updated = 111111
    depends_on = {'other_test': 222222}
    times = 0
    for r in xrange(repeats):
        all_keys = [{'date': 180101,
                     'mouse': 'TM001',
                     'run': 1,
                     'foo': str(uuid.uuid4())} for x in xrange(n)]
        start_s = time.time()
        for keys in all_keys:
            db.store(
                analysis_name='test_store_array', data=data, keys=keys,
                updated=updated, depends_on=depends_on)
        times += (time.time() - start_s) / n
    if verbose:
        print(db)
        print(" {} - {} repeats of {}".format(times / repeats, repeats, n))
    return times / repeats, n, repeats

def test_replace_array(db, verbose=True):
    n = 1000
    repeats = 3
    data = np.arange(200)
    keys = {'date': 180101,
            'mouse': 'TM001',
            'run': 1,
            'foo': 'bar'}
    updated = 111111
    depends_on = {'other_test': 222222}
    db.store(
        analysis_name='test_replace_array', data=data, keys=keys,
        updated=updated, depends_on=depends_on)
    times = 0
    for r in xrange(repeats):
        start_s = time.time()
        for _ in xrange(n):
            db.store(
                analysis_name='test_replace_array', data=data, keys=keys,
                updated=updated, depends_on=depends_on)
        times += (time.time() - start_s) / n
    if verbose:
        print(db)
        print(" {} - {} repeats of {}".format(times / repeats, repeats, n))
    return times / repeats, n, repeats

def test_recall_array(db, verbose=True):
    n = 1000
    repeats = 3
    data = np.arange(200)
    keys = {'date': 180101,
            'mouse': 'TM001',
            'run': 1,
            'foo': 'bar'}
    updated = 111111
    depends_on = {'other_test': 222222}
    db.store(
        analysis_name='test_recall_array', data=data, keys=keys,
        updated=updated, depends_on=depends_on)
    times = 0
    for r in xrange(repeats):
        start_s = time.time()
        for _ in xrange(n):
            db.recall(analysis_name='test_recall_array', keys=keys)
        times += (time.time() - start_s) / n
    if verbose:
        print(db)
        print(" {} - {} repeats of {}".format(times / repeats, repeats, n))
    return times / repeats, n, repeats

def test_store_df(db, verbose=True):
    n = 1000
    repeats = 3
    data = (pd
            .DataFrame({'value': [1, 2], 'mouse': ['TM001', 'TM001']})
            .set_index('mouse')
            )
    updated = 111111
    depends_on = {'other_test': 222222}
    times = 0
    for r in xrange(repeats):
        all_keys = [{'date': 180101,
                     'mouse': 'TM001',
                     'run': 1,
                     'foo': str(uuid.uuid4())} for x in xrange(n)]
        start_s = time.time()
        for keys in all_keys:
            db.store(
                analysis_name='test_store_df', data=data, keys=keys,
                updated=updated, depends_on=depends_on)
        times += (time.time() - start_s) / n
    if verbose:
        print(db)
        print(" {} - {} repeats of {}".format(times / repeats, repeats, n))
    return times / repeats, n, repeats


def test_replace_df(db, verbose=True):
    n = 1000
    repeats = 3
    data = (pd
            .DataFrame({'value': [1, 2], 'mouse': ['TM001', 'TM001']})
            .set_index('mouse')
            )
    keys = {'date': 180101,
            'mouse': 'TM001',
            'run': 1,
            'foo': 'bar'}
    updated = 111111
    depends_on = {'other_test': 222222}
    db.store(
        analysis_name='test_replace_df', data=data, keys=keys,
        updated=updated, depends_on=depends_on)
    times = 0
    for r in xrange(repeats):
        start_s = time.time()
        for _ in xrange(n):
            db.store(
                analysis_name='test_replace_df', data=data, keys=keys,
                updated=updated, depends_on=depends_on)
        times += (time.time() - start_s) / n
    if verbose:
        print(db)
        print(" {} - {} repeats of {}".format(times / repeats, repeats, n))
    return times / repeats, n, repeats

def test_recall_df(db, verbose=True):
    n = 1000
    repeats = 3
    data = (pd
            .DataFrame({'value': [1, 2], 'mouse': ['TM001', 'TM001']})
            .set_index('mouse')
            )
    keys = {'date': 180101,
            'mouse': 'TM001',
            'run': 1,
            'foo': 'bar'}
    updated = 111111
    depends_on = {'other_test': 222222}
    db.store(
        analysis_name='test_recall_df', data=data, keys=keys,
        updated=updated, depends_on=depends_on)
    times = 0
    for r in xrange(repeats):
        start_s = time.time()
        for _ in xrange(n):
            db.recall(analysis_name='test_recall_df', keys=keys)
        times += (time.time() - start_s) / n
    if verbose:
        print(db)
        print(" {} - {} repeats of {}".format(times / repeats, repeats, n))
    return times / repeats, n, repeats


def full_test(relative=True):
    mem_db = pool.database.db(backend='memory')
    disk_savedir = mkdtemp()
    print disk_savedir
    disk_db = pool.database.db(backend='disk', savedir=disk_savedir)
    dbs = [mem_db, disk_db]
    db_names = ['mem', 'disk']
    try:
        couch_db = pool.database.db(backend='couch', database='testing')
    except:
        pass
    else:
        dbs.append(couch_db)
        db_names.append('couch')
    mongo_db = pool.database.db(backend='mongo', collection='testing')
    dbs.append(mongo_db)
    db_names.append('mongo')

    tests = [
             # test_store_float, test_store_array, test_store_df,
             # test_replace_float, test_replace_array, test_replace_df,
             test_recall_float, test_recall_array, test_recall_df,
             ]

    for test in tests:
        print(test)
        time, n, repeats = {}, {}, {}
        for db, db_name in zip(dbs, db_names):
            time[db_name], n[db_name], repeats[db_name] = test(db, verbose=False)
        print('n = {}, repeats = {}'.format(n[db_name], repeats[db_name]))
        print('--------------------')
        if relative:
            for db_name in db_names[1:]:
                print("{}: {}x memory".format(db_name, time[db_name] / time['mem']))
        print('')

    shutil.rmtree(disk_savedir)

if __name__ == '__main__':
    full_test()
