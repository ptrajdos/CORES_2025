import os

EXPROOT = os.path.abspath(os.path.dirname(__file__))
DATAPATH = os.path.join(EXPROOT,"../","data")

EXPERIMENTS_RESULTS_PATH  = os.path.join(EXPROOT, "../","experiments_results")
EXPERIMENTS_LOGS_PATH  = os.path.join(EXPROOT, "../","experiments_logs")
EXPERIMENTS_CACHE_PATH = os.path.join(EXPROOT, "../", "experiments_cache")
