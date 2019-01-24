"""Script to add mice/dates/runs to the metadata record."""
from __future__ import print_function
from builtins import str
import argparse
import glob
import os
import os.path as opath
import pandas as pd
import re
import textwrap

import flow
import flow.metadata as fm


DEFAULTS = {
    1: {
        'run_type': 'running',
        'tags': ['hungry']},
    2: {
        'run_type': 'training',
        'tags': ['hungry']},
    3: {
        'run_type': 'training',
        'tags': ['hungry']},
    4: {
        'run_type': 'training',
        'tags': ['hungry']},
    6: {
        'run_type': 'spontaneous',
        'tags': ['sated']},
    8: {
        'run_type': 'spontaneous',
        'tags': ['sated']},
    9: {
        'run_type': 'spontaneous',
        'tags': ['sated']},
    10: {
        'run_type': 'spontaneous',
        'tags': ['sated']},
    11: {
        'run_type': 'spontaneous',
        'tags': ['sated']}
}


def add_mouse_date_runs(
        mouse, date, runs, mouse_tags=None, date_tags=None, run_tags=None,
        photometry=None, no_action=False, verbose=False):
    """Add a set of runs to the metadata record.

    Parameters
    ----------
    mouse: str
    date : int
    runs : list of int
    mouse_tags : list of str. optional
    date_tags : list of str, optional
    run_tags : list of str, optional
    photometry : list of str, optional
    no_action : bool
        If True, do everything except actually writing data; nothing will change.
    verbose : bool
        If True, report all changes that are made.

    """
    if run_tags is None:
        run_tags = []

    if not no_action:
        try:
            fm.add_mouse(mouse, tags=mouse_tags, overwrite=False)
        except fm.AlreadyPresentError:
            pass
        else:
            if verbose:
                print("Added mouse: {}, tags: {}".format(mouse, mouse_tags))
    else:
        if verbose:
            print("Adding mouse: {}, tags: {}".format(mouse, mouse_tags))

    if not no_action:
        try:
            fm.add_date(
                mouse, date, photometry=photometry, tags=date_tags,
                overwrite=False)
        except fm.AlreadyPresentError:
            pass
        else:
            if verbose:
                print("Added date: {}-{}, tags: {}".format(
                    mouse, date, date_tags))
    else:
        if verbose:
            print("Adding date: {}-{}, tags: {}".format(
                mouse, date, date_tags))

    for run in runs:
        run_type = DEFAULTS[run]['run_type']
        tags = sorted(set(DEFAULTS[run]['tags']).union(run_tags))
        if not no_action:
            try:
                fm.add_run(
                    mouse, date, run, run_type=run_type, tags=tags,
                    overwrite=False)
            except fm.AlreadyPresentError:
                if verbose:
                    print("Run already present, skipping: {}-{}-{}".format(
                        mouse, date, run))
            else:
                if verbose:
                    print("Added run: {}-{}-{}, tags: {}".format(
                        mouse, date, run, tags))
        else:
            if verbose:
                print("Adding run: {}-{}-{}, tags: {}".format(
                    mouse, date, run, tags))

def check_date(mouse, date):
    """Check date and locate if needed.

    Parameters
    ----------
    mouse : str
    date : int

    Returns
    -------
    dates : list of int

    """
    if date < 0:
        dates = locate_dates(mouse)
    else:
        dates = [date]
    return dates

def check_runs(mouse, date, runs):
    """Check runs and locate if needed.

    Parameters
    ----------
    mouse : str
    date : int
    runs : list of int

    """
    if not len(runs):
        runs = locate_runs(mouse, date)
        if not len(runs):
            raise ValueError('Unable to locate runs for {}-{}'.format(
                mouse, date))
    return runs

def locate_dates(mouse):
    """Given a mouse, locate all dates from simpcells.

    Parameters
    ----------
    mouse : str

    """
    datad = flow.config.params()['paths']['data']
    mouse_dir = opath.join(datad, mouse)
    dates = []
    for f in os.listdir(mouse_dir):
        if opath.isdir(opath.join(mouse_dir, f)):
            try:
                flow.misc.parse_date(f)
            except ValueError:
                pass
            else:
                dates.append(int(f))
    return sorted(dates)


def locate_runs(mouse, date):
    """Given a mouse and a date, locate all run simpcell files.

    Parameters
    ----------
    mouse : str
    date : int

    """
    pattern = '{}_{}_'.format(mouse, date) + \
        '(?P<run>[0-9]{3})\\.simpcell'
    datad = flow.config.params()['paths']['data']
    run_simpcells = glob.glob(opath.join(
        datad, mouse, str(date), '*.simpcell'))
    runs = []
    for simpcell in run_simpcells:
        match = re.match(pattern, opath.basename(simpcell))
        runs.append(int(match.groupdict()['run']))
    return sorted(runs)


def list_runs(
        mouse, date=None, runs=None, mouse_tags=None, date_tags=None,
        run_tags=None, photometry=None):
    """Print all runs that match the given parameters.

    Parameters
    ----------
    mouse : str
    date : int, optional
    runs : list of int, optional
    mouse_tags : list of str, optional
    date_tags : list of str, optional
    run_tags : list of str, optional
    photometry : list of str, optional

    """
    if date is None or date < 0:
        dates = None
    else:
        dates = [date]
    if runs is None or not len(runs):
        runs = None
    if mouse_tags is None:
        mouse_tags = []
    if date_tags is None:
        date_tags = []
    if run_tags is None:
        run_tags = []
    tags = mouse_tags + date_tags + run_tags

    meta = fm.meta(mice=[mouse], dates=dates, runs=runs, tags=tags,
                   photometry=photometry)

    with pd.option_context('display.max_rows', None,
                           'display.max_columns', None):
        try:
            new_meta = meta.set_index(['mouse', 'date', 'run'])
        except KeyError:
            new_meta = meta
        print(new_meta)


def main():
    """Main script."""
    arg_parser = argparse.ArgumentParser(description=textwrap.dedent("""
        Helper script to add metadata for analysis, using lab standards.
        
        Can be called several different ways.
        Specify a single mouse, single date and runs:
        > python metadata.py OA178 180701 1 2 3 4
        
        Specify a single mouse and date, infers runs from simpcells.
        > python metadata.py OA178 180701
        
        Specify a single mouse, finds all dates and runs from simpcells.
        > python metadata.py OA178
        
        List all matching runs and don't add anything.
        > python metadata.py --list OA178 180701
        """), epilog=textwrap.dedent("""
        Run types and default run tags are inferred based on our lab standards.
        Photometry and the tags can be specified multiple times to add more
        than 1 value. For example:
        > python metadata.py OA178 180701 1 2 3 9 10 11 -m jeff -m test
        """), formatter_class=argparse.RawDescriptionHelpFormatter,)
    arg_parser.add_argument(
        "mouse", action="store", help="Name of mouse to add.")
    arg_parser.add_argument(
        "date", action="store", type=int, nargs='?', default=-1,
        help="Date to add. If none passed, checks simpcells.")
    arg_parser.add_argument(
        "runs", action="store", type=int, nargs="*",
        help="Runs to add. If none passed, checks simpcells.")
    arg_parser.add_argument(
        "-p", "--photometry", action="append", default=None,
        help="Add photometry.")
    arg_parser.add_argument(
        "-m", "--mouse_tag", action="append", default=None,
        help="Tag to add to the mouse.")
    arg_parser.add_argument(
        "-d", "--date_tag", action="append", default=None,
        help="Tag to add to the date.")
    arg_parser.add_argument(
        "-r", "--run_tag", action="append", default=None,
        help="Tag to add to ALL runs. Combined with default tags.")
    arg_parser.add_argument(
        "-l", "--list", action="store_true",
        help="Only list currently matching runs, don't actually add anything.")
    arg_parser.add_argument(
        "-n", "--no_action", action="store_true",
        help="Do nothing.")
    arg_parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Be verbose.")
    args = arg_parser.parse_args()

    if args.list:
        list_runs(args.mouse, args.date, args.runs, args.mouse_tag,
                  args.date_tag, args.run_tag, args.photometry)
        return

    dates = check_date(args.mouse, args.date)

    for date in dates:
        runs = check_runs(args.mouse, date, args.runs)
        add_mouse_date_runs(
            mouse=args.mouse, date=date, runs=runs,
            mouse_tags=args.mouse_tag, date_tags=args.date_tag,
            run_tags=args.run_tag, photometry=args.photometry,
            no_action=args.no_action, verbose=args.verbose)


if __name__ == '__main__':
    main()
