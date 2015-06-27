# Copyright (C) 2014-2015 Julius Susanto. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

"""
Three-phase fault test script
"""

import numpy as np
from pyfault.runfault_3ph import runfault_3ph
from pypower.loadcase import loadcase
from pypf.casedata import convert_ppc

if __name__ == '__main__':
    # Load PYPOWER case
    ppc = convert_ppc(loadcase('case9.py'))
    ppc.gen.Rgen = np.array([0.01, 0.01, 0.005])
    ppc.gen.Xdpp = np.array([0.12, 0.16, 0.100])
    
    # Run 3ph fault calculation
    success, results = runfault_3ph(ppc)
