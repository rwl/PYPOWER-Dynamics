# Copyright (c) 1996-2015 PSERC. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

"""Checks for dispatchable loads.
"""

from numpy import zeros

def isload(gen):
    """Checks for dispatchable loads.

    Returns a column vector of 1's and 0's. The 1's correspond to rows of the
    C{gen} matrix which represent dispatchable loads. The current test is
    C{Pmin < 0 and Pmax == 0}. This may need to be revised to allow sensible
    specification of both elastic demand and pumped storage units.

    @author: Ray Zimmerman (PSERC Cornell)
    """
    return zeros(gen.n).astype(int)# (gen.Pmin < 0) & (gen.Pmax == 0)
