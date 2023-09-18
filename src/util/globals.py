U_CUTOFF = 0.05
# the higher this value, the closer to 0.0 the genetic distance is when put into e^(GENETIC_ALPHA*x)
G_IDENTICAL_CLONE_VALUE = 2.0
ORGANOTROP_ALPHA = -5.0
GENETIC_ALPHA = -5.0

import logging
logger = logging.getLogger('SGD')
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s\n\r%(message)s', datefmt='%H:%M:%S')
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)

MIG_KEY = "migration_number"
COMIG_KEY = "comigration_number"
SEEDING_KEY = "seeding_site_number"
ORGANOTROP_KEY = "organotropism"
GEN_DIST_KEY = "genetic_distance"
DATA_FIT_KEY = "neg_log_likelihood"
REG_KEY = "regularizer"
FULL_LOSS_KEY = "loss"
