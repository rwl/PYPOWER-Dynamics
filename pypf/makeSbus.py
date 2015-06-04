# Copyright (c) 1996-2015 PSERC. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

"""Builds the vector of complex bus power injections.
"""

from numpy import array, ones, flatnonzero as find
from scipy.sparse import csr_matrix as sparse


def makeSbus(baseMVA, bus, gen):
    """Builds the vector of complex bus power injections.

    Returns the vector of complex bus power injections, that is, generation
    minus load. Power is expressed in per unit.

    @see: L{makeYbus}

    @author: Ray Zimmerman (PSERC Cornell)
    """
    ## generator info
    on = find(array(gen.status) > 0)      ## which generators are on?
    gbus = array(gen.bus)[on]             ## what buses are they at?

    ## form net complex bus power injection vector
    nb = bus.n
    ngon = on.shape[0]
    ## connection matrix, element i, j is 1 if gen on(j) at bus i is ON
    Cg = sparse((ones(ngon), (gbus, range(ngon))), (nb, ngon))

    ## power injected by gens plus power injected by loads converted to p.u.
    Sbus = ( Cg * (gen.Pg[on] + 1j * gen.Qg[on]) -
             (bus.Pd + 1j * bus.Qd) ) / baseMVA

    return Sbus
