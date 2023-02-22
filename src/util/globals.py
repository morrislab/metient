U_CUTOFF = 0.05
# TODO: better way to handle this?
G_IDENTICAL_CLONE_VALUE = -10.0

import logging
logger = logging.getLogger('SGD')
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s\n\r%(message)s', datefmt='%H:%M:%S')
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)
