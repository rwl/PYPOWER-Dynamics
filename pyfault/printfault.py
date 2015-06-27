# Copyright (C) 2014-2015 Julius Susanto. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

"""
Prints fault analysis results.
"""

from sys import stdout

from numpy import \
    ones, zeros, r_, sort, exp, pi, diff, arange, min, \
    argmin, argmax, logical_or, real, imag, any, sqrt

from numpy import flatnonzero as find

from pypf.isload import isload
from pypower.ppoption import ppoption


def printfault(baseMVA, bus=None, gen=None, branch=None, f=None, success=None,
            et=None, fd=None, ppopt=None):
    """
    Prints fault analysis results.
    """
    ##----- initialization -----
    ## default arguments
    if isinstance(baseMVA, dict):
        have_results_struct = 1
        results = baseMVA
        if gen is None:
            ppopt = ppoption()   ## use default options
        else:
            ppopt = gen
        if (ppopt['OUT_ALL'] == 0):
            return     ## nothin' to see here, bail out now
        if bus is None:
            fd = stdout         ## print to stdout by default
        else:
            fd = bus
        baseMVA, bus, gen, branch, success, et = \
            results.baseMVA, results.bus, results.gen, \
            results.branch, results.success, results.et
        if 'f' in results:
            f = results["f"]
        else:
            f = None
    else:
        have_results_struct = 0
        if ppopt is None:
            ppopt = ppoption()   ## use default options
            if fd is None:
                fd = stdout         ## print to stdout by default
        if ppopt['OUT_ALL'] == 0:
            return     ## nothin' to see here, bail out now

    isOPF = f is not None    ## FALSE -> only simple PF data, TRUE -> OPF data

    ## options
    isDC            = ppopt['PF_DC']        ## use DC formulation?
    OUT_ALL         = ppopt['OUT_ALL']
    OUT_ANY         = OUT_ALL == 1     ## set to true if any pretty output is to be generated
    OUT_SYS_SUM     = (OUT_ALL == 1) or ((OUT_ALL == -1) and ppopt['OUT_SYS_SUM'])
    OUT_AREA_SUM    = (OUT_ALL == 1) or ((OUT_ALL == -1) and ppopt['OUT_AREA_SUM'])
    OUT_BUS         = (OUT_ALL == 1) or ((OUT_ALL == -1) and ppopt['OUT_BUS'])
    OUT_BRANCH      = (OUT_ALL == 1) or ((OUT_ALL == -1) and ppopt['OUT_BRANCH'])
    OUT_GEN         = (OUT_ALL == 1) or ((OUT_ALL == -1) and ppopt['OUT_GEN'])
    OUT_ANY         = OUT_ANY | ((OUT_ALL == -1) and
                        (OUT_SYS_SUM or OUT_AREA_SUM or OUT_BUS or
                         OUT_BRANCH or OUT_GEN))

    if OUT_ALL == -1:
        OUT_ALL_LIM = ppopt['OUT_ALL_LIM']
    elif OUT_ALL == 1:
        OUT_ALL_LIM = 2
    else:
        OUT_ALL_LIM = 0

    OUT_ANY         = OUT_ANY or (OUT_ALL_LIM >= 1)
    if OUT_ALL_LIM == -1:
        OUT_V_LIM       = ppopt['OUT_V_LIM']
        OUT_LINE_LIM    = ppopt['OUT_LINE_LIM']
        OUT_PG_LIM      = ppopt['OUT_PG_LIM']
        OUT_QG_LIM      = ppopt['OUT_QG_LIM']
    else:
        OUT_V_LIM       = OUT_ALL_LIM
        OUT_LINE_LIM    = OUT_ALL_LIM
        OUT_PG_LIM      = OUT_ALL_LIM
        OUT_QG_LIM      = OUT_ALL_LIM

    OUT_ANY         = OUT_ANY or ((OUT_ALL_LIM == -1) and (OUT_V_LIM or OUT_LINE_LIM or OUT_PG_LIM or OUT_QG_LIM))
    ptol = 1e-4        ## tolerance for displaying shadow prices

    ## create map of external bus numbers to bus indices
    i2e = arange(bus.n)#bus[:, BUS_I].astype(int)
    e2i = zeros(max(i2e) + 1, int)
    e2i[i2e] = arange(bus.n)

    ## sizes of things
    nb = bus.n      ## number of buses
    nl = branch.n   ## number of branches
    ng = gen.n      ## number of generators

    ## zero out some data to make printout consistent for DC case
    if isDC:
        bus.Qd = zeros(nb)
        bus.Bs = zeros(nb)
        gen.Qg = zeros(ng)
        gen.Qmax = zeros(ng)
        gen.Qmin = zeros(ng)
        branch.r = zeros(nl)
        branch.b = zeros(nl)

    ## parameters
    ties = find(bus.area[e2i[branch.f_bus]] != bus.area[e2i[branch.t_bus]])
                            ## area inter-ties
    tap = ones(nl)                           ## default tap ratio = 1 for lines
    xfmr = find(branch.tap)           ## indices of transformers
    tap[xfmr] = branch.tap[xfmr]            ## include transformer tap ratios
    tap = tap * exp(1j * pi / 180 * branch.shift) ## add phase shifters
    nzld = find((bus.Pd != 0.0) | (bus.Qd != 0.0))
    sorted_areas = sort(bus.area)
    ## area numbers
    s_areas = sorted_areas[r_[1, find(diff(sorted_areas)) + 1]]
    nzsh = find((bus.Gs != 0.0) | (bus.Bs != 0.0))
    allg = find( ~isload(gen) )
    ong  = find( (gen.status > 0) & ~isload(gen) )
    onld = find( (gen.status > 0) &  isload(gen) )
    V = bus.Vm * exp(-1j * pi / 180 * bus.Va)
    out = find(branch.status == 0)        ## out-of-service branches
    nout = len(out)
    if isDC:
        loss = zeros(nl)
    else:
        loss = baseMVA * abs(V[e2i[ branch.f_bus ]] / tap -
                             V[e2i[ branch.t_bus ]])**2 / \
                    (branch.r - 1j * branch.x)

    fchg = abs(V[e2i[ branch.f_bus ]] / tap)**2 * branch.b * baseMVA / 2
    tchg = abs(V[e2i[ branch.t_bus ]]      )**2 * branch.b * baseMVA / 2
    loss[out] = zeros(nout)
    fchg[out] = zeros(nout)
    tchg[out] = zeros(nout)

    ##----- print the stuff -----

    if OUT_SYS_SUM:
        fd.write('\n================================================================================')
        fd.write('\n|     System Summary                                                           |')
        fd.write('\n================================================================================')
        fd.write('\n\nHow many?                How much?              P (MW)            Q (MVAr)')
        fd.write('\n---------------------    -------------------  -------------  -----------------')
        fd.write('\nBuses         %6d     Total Gen Capacity   %7.1f       %7.1f to %.1f' % (nb, sum(gen.Pmax[allg]), sum(gen.Qmin[allg]), sum(gen.Qmax[allg])))
        fd.write('\nGenerators     %5d     On-line Capacity     %7.1f       %7.1f to %.1f' % (len(allg), sum(gen.Pmax[ong]), sum(gen.Qmin[ong]), sum(gen.Qmax[ong])))
        fd.write('\nCommitted Gens %5d     Generation (actual)  %7.1f           %7.1f' % (len(ong), sum(gen.Pg[ong]), sum(gen.Qg[ong])))
        fd.write('\nLoads          %5d     Load                 %7.1f           %7.1f' % (len(nzld)+len(onld), sum(bus.Pd[nzld])-sum(gen.Pg[onld]), sum(bus.Qd[nzld])-sum(gen.Qg[onld])))
        fd.write('\n  Fixed        %5d       Fixed              %7.1f           %7.1f' % (len(nzld), sum(bus.Pd[nzld]), sum(bus.Qd[nzld])))
        fd.write('\n  Dispatchable %5d       Dispatchable       %7.1f of %-7.1f%7.1f' % (len(onld), -sum(gen.Pg[onld]), -sum(gen.Pmin[onld]), -sum(gen.Qg[onld])))
        fd.write('\nShunts         %5d     Shunt (inj)          %7.1f           %7.1f' % (len(nzsh),
            -sum(bus.Vm[nzsh]**2 * bus.Gs[nzsh]), sum(bus.Vm[nzsh]**2 * bus.Bs[nzsh]) ))
        fd.write('\nBranches       %5d     Losses (I^2 * Z)     %8.2f          %8.2f' % (nl, sum(loss.real), sum(loss.imag) ))
        fd.write('\nTransformers   %5d     Branch Charging (inj)     -            %7.1f' % (len(xfmr), sum(fchg) + sum(tchg) ))
        fd.write('\nInter-ties     %5d     Total Inter-tie Flow %7.1f           %7.1f' % (len(ties), sum(abs(branch.Pf[ties]-branch.Pt[ties])) / 2, sum(abs(branch.Qf[ties]-branch.Qt[ties])) / 2))
        fd.write('\nAreas          %5d' % len(s_areas))
        fd.write('\n')
 
    ## bus data
    if OUT_BUS:
        fd.write('\n============================================')
        fd.write('\n|     Bus Data                             |')
        fd.write('\n============================================')
        fd.write('\n Bus      Nom Volt     Three-Phase Fault')
        fd.write('\n  #         kV          MVA         kA')
        fd.write('\n--------------------------------------------')
        for i in range(nb):
            fd.write('\n%3d %11.1f %14.2f %9.2f' % (i, bus.baseKV[i], bus.faultMVA[i], bus.faultMVA[i] / sqrt(3) / bus.baseKV[i]))
        
        fd.write('\n')
