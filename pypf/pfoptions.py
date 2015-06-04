
from dpower_pb2 import NEWTON

def init_pf_options(opts):
    opts.dc = False
    opts.algorithm = NEWTON
    opts.tolerance = 1e-8
    opts.iterMax = 10
    opts.enforceQLimits = False
