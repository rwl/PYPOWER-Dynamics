# Copyright (c) 1996-2015 PSERC. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

"""Updates bus, gen, branch data structures to match power flow soln.
"""

from numpy import array, asarray, angle, pi, conj, zeros, ones, finfo
from numpy import flatnonzero as find

from scipy.sparse import csr_matrix

EPS = finfo(float).eps


def pfsoln(baseMVA, bus0, gen0, branch0, Ybus, Yf, Yt, V, ref, pv, pq):
    """Updates bus, gen, branch data structures to match power flow soln.

    @author: Ray Zimmerman (PSERC Cornell)
    """
    ## initialize return values
    bus     = bus0
    gen     = gen0
    branch  = branch0

    ##----- update bus voltages -----
    bus.Vm = abs(V)
    bus.Va = angle(V) * 180 / pi

    ##----- update Qg for all gens and Pg for slack bus(es) -----
    ## generator info
    on = find(gen.status > 0) ## which generators are on?
    gbus = gen.bus[on].astype(int)  ## what buses are they at?

    ## compute total injected bus powers
    Sbus = V[gbus] * conj(Ybus[gbus, :] * V)

    ## update Qg for all generators
    gen.Qg = zeros(gen.n)              ## zero out all Qg
    gen.Qg[on] = Sbus.imag * baseMVA + bus.Qd[gbus]    ## inj Q + local Qd
    ## ... at this point any buses with more than one generator will have
    ## the total Q dispatch for the bus assigned to each generator. This
    ## must be split between them. We do it first equally, then in proportion
    ## to the reactive range of the generator.

    if len(on) > 1:
        ## build connection matrix, element i, j is 1 if gen on(i) at bus j is ON
        nb = bus.n
        ngon = on.shape[0]
        Cg = csr_matrix((ones(ngon), (range(ngon), gbus)), (ngon, nb))

        ## divide Qg by number of generators at the bus to distribute equally
        ngg = Cg * Cg.sum(0).T    ## ngon x 1, number of gens at this gen's bus
        ngg = asarray(ngg).flatten()  # 1D array
        gen.Qg[on] = gen.Qg[on] / ngg

        ## divide proportionally
        Cmin = csr_matrix((gen.Qmin[on], (range(ngon), gbus)), (ngon, nb))
        Cmax = csr_matrix((gen.Qmax[on], (range(ngon), gbus)), (ngon, nb))
        Qg_tot = Cg.T * gen.Qg[on]## nb x 1 vector of total Qg at each bus
        Qg_min = Cmin.sum(0).T       ## nb x 1 vector of min total Qg at each bus
        Qg_max = Cmax.sum(0).T       ## nb x 1 vector of max total Qg at each bus
        Qg_min = asarray(Qg_min).flatten()  # 1D array
        Qg_max = asarray(Qg_max).flatten()  # 1D array
        ## gens at buses with Qg range = 0
        ig = find(Cg * Qg_min == Cg * Qg_max)
        Qg_save = gen.Qg[on[ig]]
        gen.Qg[on] = gen.Qmin[on] + \
            (Cg * ((Qg_tot - Qg_min) / (Qg_max - Qg_min + EPS))) * \
                (gen.Qmax[on] - gen.Qmin[on])    ##    ^ avoid div by 0
        gen.Qg[on[ig]] = Qg_save  ## (terms are mult by 0 anyway)

    ## update Pg for slack bus(es)
    ## inj P + local Pd
    for k in range(len(ref)):
        refgen = find(gbus == ref[k])  ## which is(are) the reference gen(s)?
        gen.Pg[on[refgen[0]]] = \
                Sbus[refgen[0]].real * baseMVA + bus.Pd[ref[k]]
        if len(refgen) > 1:       ## more than one generator at this ref bus
            ## subtract off what is generated by other gens at this bus
            gen.Pg[on[refgen[0]]] = \
                gen.Pg[on[refgen[0]]] - sum(gen.Pg[on[refgen[1:len(refgen)]]])

    ##----- update/compute branch power flows -----
    out = find(array(branch.status) == 0)        ## out-of-service branches
    br =  find(branch.status).astype(int) ## in-service branches

    ## complex power at "from" bus
    Sf = V[ branch.f_bus[br].astype(int) ] * conj(Yf[br, :] * V) * baseMVA
    ## complex power injected at "to" bus
    St = V[ branch.t_bus[br].astype(int) ] * conj(Yt[br, :] * V) * baseMVA

    branch.Pf[br] = Sf.real
    branch.Qf[br] = Sf.imag
    branch.Pt[br] = St.real
    branch.Qt[br] = St.imag
    
    branch.Pf[out] = zeros(len(out))
    branch.Qf[out] = zeros(len(out))
    branch.Pt[out] = zeros(len(out))
    branch.Qt[out] = zeros(len(out))

    return bus, gen, branch
