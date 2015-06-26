# Copyright (c) 1996-2015 PSERC. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

"""Builds the FDPF matrices, B prime and B double prime.
"""

from numpy import ones, zeros

from pypf.makeYbus import makeYbus


def makeB(baseMVA, bus, branch, alg):
    """Builds the FDPF matrices, B prime and B double prime.

    Returns the two matrices B prime and B double prime used in the fast
    decoupled power flow. Does appropriate conversions to p.u. C{alg} is the
    value of the C{PF_ALG} option specifying the power flow algorithm.

    @see: L{fdpf}

    @author: Ray Zimmerman (PSERC Cornell)
    """
    ## constants
    nb = bus.n          ## number of buses
    nl = branch.n       ## number of lines

    ##-----  form Bp (B prime)  -----
    temp_branch = branch.copy()                 ## modify a copy of branch
    temp_bus = bus.copy()                       ## modify a copy of bus
    temp_bus.bs = zeros(nb)                ## zero out shunts at buses
    temp_branch.b = zeros(nl)           ## zero out line charging shunts
    temp_branch.tap = ones(nl)             ## cancel out taps
    if alg == 2:                               ## if XB method
        temp_branch.r = zeros(nl)       ## zero out line resistance
    Bp = -1 * makeYbus(baseMVA, temp_bus, temp_branch)[0].imag

    ##-----  form Bpp (B double prime)  -----
    temp_branch = branch.copy()                 ## modify a copy of branch
    temp_branch.shift = zeros(nl)          ## zero out phase shifters
    if alg == 3:                               ## if BX method
        temp_branch.r = zeros(nl)    ## zero out line resistance
    Bpp = -1 * makeYbus(baseMVA, bus, temp_branch)[0].imag

    return Bp, Bpp
