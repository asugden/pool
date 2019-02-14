"""Configurable defaults for pool."""
from __future__ import print_function
from builtins import input
import copy
import json
import os
import shutil

CONFIG_FILE = 'pool.cfg'
DEFAULT_FILE = 'pool.cfg.default'
CONFIG_PATHS = [
    os.path.expanduser('~/.config/pool'),
    os.path.join(os.path.dirname(__file__)),
    os.environ.get("POOL_CONF"),
]
supported_backends = \
    ["memory", "shelve", "couch", "disk", "null"]

_stimuli = ['plus', 'neutral', 'minus']

_colors = {
    'plus': '#47D1A8',  # mint
    'neutral': '#47AEED',  # blue
    'minus': '#D61E21',  # red
    'plus-neutral': '#8CD7DB',  # aqua
    'plus-minus': '#F2E205',  # yellow
    'neutral-minus': '#C880D1',  # purple
    'other': '#7C7C7C',  # gray
    'other-running': '#333333',  # dark gray
    'pavlovian': '#47D1A8',  # green
    'real': '#47D1A8',  # yellow
    'circshift': '#8CD7DB',  # aqua
    'run-onset': '#C880D1',  # purple
    'motion-onset': '#D61E21',  # red
    'popact': '#333333',  # dark gray
    'disengaged1': '#F2E205',  # yellow
    'disengaged2': '#E86E0A',  # orange
    'ensure': '#5E5AE6',  # indigo
    'quinine': '#E86E0A',  # orange

    'plus-only': '#47D1A8',  # mint
    'neutral-only': '#47AEED',  # blue
    'minus-only': '#D61E21',  # red
    'ensure-only': '#5E5AE6',  # indigo
    'quinine-only': '#E86E0A',  # orange
    'ensure-multiplexed': '#5E5AE6',  # indigo
    'quinine-multiplexed': '#E86E0A',  # orange
    'plus-ensure': '#5E5AE6',  # indigo
    'minus-quinine': '#E86E0A',  # orange

    'lick': '#F2E205',  # yellow
    'undefined': '#7C7C7C',  # gray
    'multiplexed': '#000000',  # black
    'combined': '#000000',  # black
    'temporal-prior': '#C880D1',  # purple

    'inhibited': '#7C7C7C',  # gray

    'reward-cluster-1': '#5E5AE6',  # indigo
    'reward-cluster-exclusive-1': '#C880D1',  # purple
    'reward': '#C880D1',  # purple
    'non': '#E86E0A',  # orange

    'orange': '#E86E0A',
    'red': '#D61E21',
    'gray': '#7C7C7C',
    'black': '#000000',
    'green': '#75D977',
    'mint': '#47D1A8',
    'purple': '#C880D1',
    'indigo': '#5E5AE6',
    'blue': '#47AEED',  # previously 4087DD
    'yellow': '#F2E205',
}

_params = None


def params(reload_=False):
    """Return a copy of the parameters dictionary.

    This is the primary function that should be used to access user-specific
    parameters.

    For 'defaults' and 'colors', they are initialized with the default values
    in this file, but overwritten by any settings in the user's config file.

    Parameters
    ----------
    reload_ : bool
        If True, reload the config file from disk.

    """
    global _params
    if reload_ or _params is None:
        _params = _load_config()
    return copy.deepcopy(_params)


def reconfigure():
    """Re-set user-configurable parameters."""
    config_path = _find_config()
    if config_path is None:
        config_path = _initialize_config()

    print("Reconfiguring pool: {}".format(config_path))
    with open(config_path, 'r') as f:
        config = json.load(f)

    config['backends']['supported_backends'] = supported_backends

    print('ANALYSIS BACKEND')
    print(' memory: stores values in memory, not persistent')
    print(' shelve: stores values in a shelve database file')
    print(' couch: store values in a CouchDB (should already be running)')
    print(' disk: store values to disk as individual files')
    print(' null: does not store anything')

    backend = None
    while backend not in config['backends']['supported_backends']:
        backend = input(
            'Enter backend type: [{}] '.format(
                config['backends']['backend']))
        if not len(backend):
            backend = config['backends']['backend']
    config['backends']['backend'] = backend

    if backend == 'couch':
        if 'couch_options' not in config['backends']:
            config['backends']['couch_options'] = {}
        host = input("Enter ip or hostname of CouchDB: [{}] ".format(
            config['backends']['couch_options'].get('host', None)))
        if len(host):
            config['backends']['couch_options']['host'] = host
        port = input("Enter port for CouchDB: [{}] ".format(
            config['backends']['couch_options'].get('port', None)))
        if len(port):
            config['backends']['couch_options']['port'] = port
        database = input("Enter name of analysis database: [{}] ".format(
            config['backends']['couch_options'].get('database', None)))
        if len(database):
            config['backends']['couch_options']['database'] = database
        user = input("Enter username to authenticate with CouchDB (optional): [{}] ".format(
            config['backends']['couch_options'].get('user', None)))
        if len(user):
            config['backends']['couch_options']['user'] = user
        password = input("Enter password to authenticate with CouchDB (optional): [{}] ".format(
            config['backends']['couch_options'].get('password', None)))
        if len(password):
            config['backends']['couch_options']['password'] = password

    large_backend = None
    while large_backend not in config['backends']['supported_backends']:
        large_backend = input(
            'Enter backend type for large files: [{}] '.format(
                config['backends'].get('large_backend', 'null')))
        if not len(large_backend):
            large_backend = config['backends'].get('backend', 'null')
    config['backends']['large_backend'] = large_backend

    with open(config_path, 'w') as f:
        json.dump(config, f, sort_keys=True, indent=4, separators=(',', ': '))

    params(reload_=True)


def color(clr=None):
    """Return default color pairings.

    Parameters
    ----------
    clr : str, optional
        If not None, return the default colors for a specific group.
        Otherwise return all color pairings.

    """
    p = params()
    if clr is None:
        return p['colors']
    else:
        return p['colors'].get(clr, '#7C7C7C')


def colors():
    return color()


def stimuli():
    """Return the default stimuli."""

    p = params()
    return p['stimuli']


def _load_config():
    config_path = _find_config()
    if config_path is None:
        config_path = _initialize_config()
        print("Configuration initialized to: {}".format(config_path))
        print("Run `import pool.config as cfg; cfg.reconfigure()` to update.")
    with open(config_path, 'r') as f:
        loaded_config = json.load(f)
        config = {'colors': copy.copy(_colors), 'stimuli': copy.copy(_stimuli)}
        for key in loaded_config:
            # Just add keys in the config file other than 'colors'
            if key not in config:
                config[key] = loaded_config[key]
            else:
                # This assumes that these keys will also contain dicts,
                # they should.
                config[key].update(loaded_config[key])
    return config


def _find_config():
    for path in CONFIG_PATHS:
        if path is None:
            continue
        if os.path.isfile(os.path.join(path, CONFIG_FILE)):
            return os.path.join(path, CONFIG_FILE)
    return None


def _initialize_config():
    for path in CONFIG_PATHS:
        if path is None:
            continue
        if not os.path.isdir(path):
            try:
                os.makedirs(path)
            except OSError:
                continue
        f = os.path.join(os.path.dirname(__file__), DEFAULT_FILE)
        try:
            shutil.copy(f, os.path.join(path, CONFIG_FILE))
            return os.path.join(path, CONFIG_FILE)
        except IOError:
            continue
    print("Unable to find writable location.")
    return DEFAULT_FILE

if __name__ == '__main__':
    params()
# from pudb import set_trace; set_trace()
