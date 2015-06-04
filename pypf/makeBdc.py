# Copyright (c) 1996-2015 PSERC. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

"""Builds the B matrices and phase shift injections for DC power flow.
"""

from numpy import ones, r_, pi, flatnonzero as find
from scipy.sparse import csr_matrix as sparse


def makeBdc(baseMVA, bus, branch):
    """Builds the B matrices and phase shift injections for DC power flow.

    Returns the B matrices and phase shift injection vectors needed for a
    DC power flow.
    The bus real power injections are related to bus voltage angles by::
        P = Bbus * Va + PBusinj
    The real power flows at the from end the lines are related to the bus
    voltage angles by::
        Pf = Bf * Va + Pfinj
    Does appropriate conversions to p.u.

    @see: L{dcpf}

    @author: Carlos E. Murillo-Sanchez (PSERC Cornell & Universidad
    Autonoma de Manizales)
    @author: Ray Zimmerman (PSERC Cornell)
    """
    ## constants
    nb = bus.n          ## number of buses
    nl = branch.n       ## number of lines

    ## check that bus numbers are equal to indices to bus (one set of bus nums)
#     if any(bus[:, BUS_I] != list(range(nb))):
#         stderr.write('makeBdc: buses must be numbered consecutively in '
#                      'bus matrix\n')

    ## for each branch, compute the elements of the branch B matrix and the phase
    ## shift "quiescent" injections, where
    ##
    ##      | Pf |   | Bff  Bft |   | Vaf |   | Pfinj |
    ##      |    | = |          | * |     | + |       |
    ##      | Pt |   | Btf  Btt |   | Vat |   | Ptinj |
    ##
    stat = branch.status               ## ones at in-service branches
    b = stat / branch.x                ## series susceptance
    tap = ones(nl)                     ## default tap ratio = 1
    i = find(branch.tap)               ## indices of non-zero tap ratios
    tap[i] = branch.tap[i]             ## assign non-zero tap ratios
    b = b / tap

    ## build connection matrix Cft = Cf - Ct for line and from - to buses
    f = branch.f_bus                           ## list of "from" buses
    t = branch.t_bus                           ## list of "to" buses
    i = r_[range(nl), range(nl)]                   ## double set of row indices
    ## connection matrix
    Cft = sparse((r_[ones(nl), -ones(nl)], (i, r_[f, t])), (nl, nb))

    ## build Bf such that Bf * Va is the vector of real branch powers injected
    ## at each branch's "from" bus
    Bf = sparse((r_[b, -b], (i, r_[f, t])))## = spdiags(b, 0, nl, nl) * Cft

    ## build Bbus
    Bbus = Cft.T * Bf

    ## build phase shift injection vectors
    Pfinj = b * (-branch.shift * pi / 180)  ## injected at the from bus ...
    # Ptinj = -Pfinj                            ## and extracted at the to bus
    Pbusinj = Cft.T * Pfinj                ## Pbusinj = Cf * Pfinj + Ct * Ptinj

    return Bbus, Bf, Pbusinj, Pfinj
