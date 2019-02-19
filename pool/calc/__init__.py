"""
Calc init.

All calc functions must be imported on initialization to allow the memoizer
to build a dictionary of update dates.

"""
import os
import glob
__all__ = [os.path.basename(f)[:-3] for f in glob.glob(os.path.dirname(__file__)+"/*.py")]
from . import *
