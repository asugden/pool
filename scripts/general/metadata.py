"""Script to add mice/dates/runs to the metadata record."""
import argparse
import glob
import os.path as opath
import re

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
        photometry=None):
    """Add a set of runs to the metadata record.

    Parameters
    ----------
    mouse: str
    date : int
    runa : list of int
    mouse_tags : optional, list of str
    date_tags : optional list of str
    run_tags : optional, list of str
    photometry : optional, list of str

    """
    if run_tags is None:
        run_tags = []

    try:
        fm.add_mouse(mouse, tags=mouse_tags, overwrite=False)
    except fm.AlreadyPresentError:
        pass

    try:
        fm.add_date(
            mouse, date, photometry=photometry, tags=date_tags,
            overwrite=False)
    except fm.AlreadyPresentError:
        pass

    for run in runs:
        run_type = DEFAULTS[run]['run_type']
        tags = sorted(set(DEFAULTS[run]['tags']).union(run_tags))
        fm.add_run(
            mouse, date, run, run_type=run_type, tags=tags, overwrite=False)


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


def main():
    """Main script."""
    arg_parser = argparse.ArgumentParser(description="""
        Helper script to add metadata for analysis, using lab standards.
        """, epilog="""
        Run types and default run tags are inferred based on our lab standards.
        Photometry and the tags can be specified multiple times to add more
        than 1 value. For example:
        `python metadata.py OA178 180701 1 2 3 9 10 11 -m jeff -m test`.
        """)
    arg_parser.add_argument(
        "mouse", action="store", help="Name of mouse to add.")
    arg_parser.add_argument(
        "date", action="store", type=int, help="Date to add.")
    arg_parser.add_argument(
        "runs", action="store", type=int, nargs="*",
        help="Runs to add. If none passed, checks simpcells.")
    arg_parser.add_argument(
        "-p", "--photometry", action="append", default=None,
        help="Add photometry.")
    arg_parser.add_argument(
        "-m", "--mouse_tags", action="append", default=None,
        help="Tags to add to the mouse.")
    arg_parser.add_argument(
        "-d", "--date_tags", action="append", default=None,
        help="Tags to add to the date.")
    arg_parser.add_argument(
        "-r", "--run_tags", action="append", default=None,
        help="Tags to add to ALL runs. Combined with default tags.")
    args = arg_parser.parse_args()

    runs = check_runs(args.mouse, args.date, args.runs)

    add_mouse_date_runs(
        mouse=args.mouse, date=args.date, runs=runs,
        mouse_tags=args.mouse_tags, date_tags=args.date_tags,
        run_tags=args.run_tags)


if __name__ == '__main__':
    main()
