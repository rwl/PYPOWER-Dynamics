# Copyright (c) 1996-2015 PSERC. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

"""Builds the vector of complex bus power injections.
"""

from numpy import zeros, ones, flatnonzero as find
import numpy as np
from scipy.sparse import csr_matrix as sparse


def makeSbus(baseMVA, bus, load, gen):
    """Builds the vector of complex bus power injections.

    Returns the vector of complex bus power injections, that is, generation
    minus load. Power is expressed in per unit.

    @see: L{makeYbus}

    @author: Ray Zimmerman (PSERC Cornell)
    """
    Sg = makeSg(bus.n, gen)
    Sd = makeSd(bus.n, load)

    # Form net complex bus power injection vector.
    return (Sg - Sd) / baseMVA


def makeSg(nb, gen):
    on = find(gen.status > 0)
    gbus = gen.bus[on]
    ngon = len(on)
    if ngon == 0:
        return zeros(nb, dtype='complex')
  
    # Connection matrix, element i, j is 1 if gen on(j) at bus i is ON.
    Cg = sparse((ones(ngon), (gbus, range(ngon))), (nb, ngon))

    Pg = gen.pg[on]
    Qg = gen.qg[on]
    Sg = np.complex(Pg, Qg);

    return Cg * Sg # power injected by generators


def makeSd(nb, load):
    on = find(load.status > 0)
    ldbus = load.bus[on]
    non = len(on)
    if non == 0:
        return zeros(nb, dtype='complex')

    Cd = sparse((ones(non), (ldbus, range(non))), (nb, non))

    Pd = load.Pd[on]
    Qd = load.Qd[on]
    Sd = np.complex(Pd, Qd)

    return Cd * Sd


def makeYsh(nb, shunt):
    on = find(shunt.status > 0)
    shbus = shunt.bus[on]
    non = len(on)
    if non == 0:
        return zeros(nb, dtype='complex')

    Csh = sparse((ones(non), (shbus, range(non))), (nb, non))

    Gs = shunt.Gs[on]
    Bs = shunt.Bs[on]
    Ysh = np.complex(Gs, Bs)

    return Csh * Ysh
