# Copyright (c) 1996-2015 PSERC. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

"""Prints power flow results.
"""

from sys import stdout

from numpy import \
    ones, zeros, r_, sort, exp, pi, diff, arange, min, \
    argmin, argmax, logical_or, real, imag, any

from numpy import flatnonzero as find

from pypf.isload import isload
from pypower.run_userfcn import run_userfcn
from pypower.ppoption import ppoption

from pypower.idx_bus import REF


def printpf(results, fd=None, ppopt=None):
    """Prints power flow results.

    Prints power flow and optimal power flow results to C{fd} (a file
    descriptor which defaults to C{stdout}), with the details of what
    gets printed controlled by the optional C{ppopt} argument, which is a
    PYPOWER options vector (see L{ppoption} for details).

    The data can either be supplied in a single C{results} dict, or
    in the individual arguments: C{baseMVA}, C{bus}, C{gen}, C{branch}, C{f},
    C{success} and C{et}, where C{f} is the OPF objective function value,
    C{success} is C{True} if the solution converged and C{False} otherwise,
    and C{et} is the elapsed time for the computation in seconds. If C{f} is
    given, it is assumed that the output is from an OPF run, otherwise it is
    assumed to be a simple power flow run.

    Examples::
        ppopt = ppoptions(OUT_GEN=1, OUT_BUS=0, OUT_BRANCH=0)
        fd = open(fname, 'w+b')
        results = runopf(ppc)
        printpf(results)
        printpf(results, fd)
        printpf(results, fd, ppopt)
        printpf(baseMVA, bus, gen, branch, f, success, et)
        printpf(baseMVA, bus, gen, branch, f, success, et, fd)
        printpf(baseMVA, bus, gen, branch, f, success, et, fd, ppopt)
        fd.close()

    @author: Ray Zimmerman (PSERC Cornell)
    """
    ##----- initialization -----
    ## default arguments
#     if isinstance(baseMVA, dict):
    have_results_struct = 1
#     results = baseMVA
    if ppopt is None:
        ppopt = ppoption()   ## use default options
#     else:
#         ppopt = gen
    if (ppopt['OUT_ALL'] == 0):
        return     ## nothin' to see here, bail out now
    if fd is None:
        fd = stdout         ## print to stdout by default
#     else:
#         fd = bus
    baseMVA, bus, gen, branch, success, et = \
        results.baseMVA, results.bus, results.gen, results.branch, \
        getattr(results, "success"), getattr(results, "et")
    if hasattr(results, 'f'):
        f = results.f
    else:
        f = None
#     else:
#         have_results_struct = 0
#         if ppopt is None:
#             ppopt = ppoption()   ## use default options
#             if fd is None:
#                 fd = stdout         ## print to stdout by default
#         if ppopt['OUT_ALL'] == 0:
#             return     ## nothin' to see here, bail out now

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
    if OUT_ANY:
        ## convergence & elapsed time
        if success:
            fd.write('\nConverged in %.2f seconds' % et)
        else:
            fd.write('\nDid not converge (%.2f seconds)\n' % et)

        ## objective function value
        if isOPF:
            fd.write('\nObjective Function Value = %.2f $/hr' % f)

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
        fd.write('\n                          Minimum                      Maximum')
        fd.write('\n                 -------------------------  --------------------------------')
        minv = min(bus.Vm)
        mini = argmin(bus.Vm)
        maxv = max(bus.Vm)
        maxi = argmax(bus.Vm)
        fd.write('\nVoltage Magnitude %7.3f p.u. @ bus %-4d     %7.3f p.u. @ bus %-4d' % (minv, arange(nb)[mini], maxv, arange(nb)[maxi]))
        minv = min(bus.Va)
        mini = argmin(bus.Va)
        maxv = max(bus.Va)
        maxi = argmax(bus.Va)
        fd.write('\nVoltage Angle   %8.2f deg   @ bus %-4d   %8.2f deg   @ bus %-4d' % (minv, arange(nb)[mini], maxv, arange(nb)[maxi]))
        if not isDC:
            maxv = max(loss.real)
            maxi = argmax(loss.real)
            fd.write('\nP Losses (I^2*R)             -              %8.2f MW    @ line %d-%d' % (maxv, branch.f_bus[maxi], branch.t_bus[maxi]))
            maxv = max(loss.imag)
            maxi = argmax(loss.imag)
            fd.write('\nQ Losses (I^2*X)             -              %8.2f MVAr  @ line %d-%d' % (maxv, branch.f_bus[maxi], branch.t_bus[maxi]))
        if isOPF:
            minv = min(bus.lam_p)
            mini = argmin(bus.lam_p)
            maxv = max(bus.lam_p)
            maxi = argmax(bus.lam_p)
            fd.write('\nLambda P        %8.2f $/MWh @ bus %-4d   %8.2f $/MWh @ bus %-4d' % (minv, arange(nb)[mini], maxv, arange(nb)[maxi]))
            minv = min(bus.lam_q)
            mini = argmin(bus.lam_q)
            maxv = max(bus.lam_q)
            maxi = argmax(bus.lam_q)
            fd.write('\nLambda Q        %8.2f $/MWh @ bus %-4d   %8.2f $/MWh @ bus %-4d' % (minv, arange(nb)[mini], maxv, arange(nb)[maxi]))
        fd.write('\n')

    if OUT_AREA_SUM:
        fd.write('\n================================================================================')
        fd.write('\n|     Area Summary                                                             |')
        fd.write('\n================================================================================')
        fd.write('\nArea  # of      # of Gens        # of Loads         # of    # of   # of   # of')
        fd.write('\n Num  Buses   Total  Online   Total  Fixed  Disp    Shunt   Brchs  Xfmrs   Ties')
        fd.write('\n----  -----   -----  ------   -----  -----  -----   -----   -----  -----  -----')
        for i in range(len(s_areas)):
            a = s_areas[i]
            ib = find(bus.area == a)
            ig = find((bus.area[e2i[gen.bus]] == a) & ~isload(gen))
            igon = find((bus.area[e2i[gen.bus]] == a) & (gen.status > 0) & ~isload(gen))
            ildon = find((bus.area[e2i[gen.bus]] == a) & (gen.status > 0) & isload(gen))
            inzld = find((bus.area == a) & logical_or(bus.Pd, bus.Qd))
            inzsh = find((bus.area == a) & logical_or(bus.Gs, bus.Bs))
            ibrch = find((bus.area[e2i[branch.f_bus]] == a) & (bus.area[e2i[branch.t_bus]] == a))
            in_tie = find((bus.area[e2i[branch.f_bus]] == a) & (bus.area[e2i[branch.t_bus]] != a))
            out_tie = find((bus.area[e2i[branch.f_bus]] != a) & (bus.area[e2i[branch.t_bus]] == a))
            if not any(xfmr + 1):
                nxfmr = 0
            else:
                nxfmr = len(find((bus.area[e2i[branch.f_bus[xfmr]]] == a) & (bus.area[e2i[branch.t_bus[xfmr]]] == a)))
            fd.write('\n%3d  %6d   %5d  %5d   %5d  %5d  %5d   %5d   %5d  %5d  %5d' %
                (a, len(ib), len(ig), len(igon), \
                len(inzld)+len(ildon), len(inzld), len(ildon), \
                len(inzsh), len(ibrch), nxfmr, len(in_tie)+len(out_tie)))

        fd.write('\n----  -----   -----  ------   -----  -----  -----   -----   -----  -----  -----')
        fd.write('\nTot: %6d   %5d  %5d   %5d  %5d  %5d   %5d   %5d  %5d  %5d' %
            (nb, len(allg), len(ong), len(nzld)+len(onld),
            len(nzld), len(onld), len(nzsh), nl, len(xfmr), len(ties)))
        fd.write('\n')
        fd.write('\nArea      Total Gen Capacity           On-line Gen Capacity         Generation')
        fd.write('\n Num     MW           MVAr            MW           MVAr             MW    MVAr')
        fd.write('\n----   ------  ------------------   ------  ------------------    ------  ------')
        for i in range(len(s_areas)):
            a = s_areas[i]
            ig = find((bus.area[e2i[gen.bus]] == a) & ~isload(gen))
            igon = find((bus.area[e2i[gen.bus]] == a) & (gen.status > 0) & ~isload(gen))
            fd.write('\n%3d   %7.1f  %7.1f to %-7.1f  %7.1f  %7.1f to %-7.1f   %7.1f %7.1f' %
                (a, sum(gen.Pmax[ig]), sum(gen.Qmin[ig]), sum(gen.Qmax[ig]),
                sum(gen.Pmax[igon]), sum(gen.Qmin[igon]), sum(gen.Qmax[igon]),
                sum(gen.Pg[igon]), sum(gen.Qg[igon]) ))

        fd.write('\n----   ------  ------------------   ------  ------------------    ------  ------')
        fd.write('\nTot:  %7.1f  %7.1f to %-7.1f  %7.1f  %7.1f to %-7.1f   %7.1f %7.1f' %
                (sum(gen.Pmax[allg]), sum(gen.Qmin[allg]), sum(gen.Qmax[allg]),
                sum(gen.Pmax[ong]), sum(gen.Qmin[ong]), sum(gen.Qmax[ong]),
                sum(gen.Pg[ong]), sum(gen.Qg[ong]) ))
        fd.write('\n')
        fd.write('\nArea    Disp Load Cap       Disp Load         Fixed Load        Total Load')
        fd.write('\n Num      MW     MVAr       MW     MVAr       MW     MVAr       MW     MVAr')
        fd.write('\n----    ------  ------    ------  ------    ------  ------    ------  ------')
        Qlim = (gen.Qmin == 0) * gen.Qmax + (gen.Qmax == 0) * gen.Qmin
        for i in range(len(s_areas)):
            a = s_areas[i]
            ildon = find((bus.area[e2i[gen.bus]] == a) & (gen.status > 0) & isload(gen))
            inzld = find((bus.area == a) & logical_or(bus.Pd, bus.Qd))
            fd.write('\n%3d    %7.1f %7.1f   %7.1f %7.1f   %7.1f %7.1f   %7.1f %7.1f' %
                (a, -sum(gen.Pmin[ildon]),
                -sum(Qlim[ildon]),
                -sum(gen.Pg[ildon]), -sum(gen.Qg[ildon]),
                sum(bus.Pd[inzld]), sum(bus.Qd[inzld]),
                -sum(gen.Pg[ildon]) + sum(bus.Pd[inzld]),
                -sum(gen.Qg[ildon]) + sum(bus.Qd[inzld]) ))

        fd.write('\n----    ------  ------    ------  ------    ------  ------    ------  ------')
        fd.write('\nTot:   %7.1f %7.1f   %7.1f %7.1f   %7.1f %7.1f   %7.1f %7.1f' %
                (-sum(gen.Pmin[onld]),
                -sum(Qlim[onld]),
                -sum(gen.Pg[onld]), -sum(gen.Qg[onld]),
                sum(bus.Pd[nzld]), sum(bus.Qd[nzld]),
                -sum(gen.Pg[onld]) + sum(bus.Pd[nzld]),
                -sum(gen.Qg[onld]) + sum(bus.Qd[nzld])) )
        fd.write('\n')
        fd.write('\nArea      Shunt Inj        Branch      Series Losses      Net Export')
        fd.write('\n Num      MW     MVAr     Charging      MW     MVAr       MW     MVAr')
        fd.write('\n----    ------  ------    --------    ------  ------    ------  ------')
        for i in range(len(s_areas)):
            a = s_areas[i]
            inzsh   = find((bus.area == a) & logical_or(bus.Gs, bus.Bs))
            ibrch   = find((bus.area[e2i[branch.f_bus]] == a) & (bus.area[e2i[branch.t_bus]] == a) & branch.status.astype(bool))
            in_tie  = find((bus.area[e2i[branch.f_bus]] != a) & (bus.area[e2i[branch.t_bus]] == a) & branch.status.astype(bool))
            out_tie = find((bus.area[e2i[branch.f_bus]] == a) & (bus.area[e2i[branch.t_bus]] != a) & branch.status.astype(bool))
            fd.write('\n%3d    %7.1f %7.1f    %7.1f    %7.2f %7.2f   %7.1f %7.1f' %
                (a, -sum(bus.Vm[inzsh]**2 * bus.Gs[inzsh]),
                 sum(bus.Vm[inzsh]**2 * bus.Bs[inzsh]),
                 sum(fchg[ibrch]) + sum(tchg[ibrch]) + sum(fchg[out_tie]) + sum(tchg[in_tie]),
                 sum(real(loss[ibrch])) + sum(real(loss[r_[in_tie, out_tie]])) / 2,
                 sum(imag(loss[ibrch])) + sum(imag(loss[r_[in_tie, out_tie]])) / 2,
                 sum(branch.Pt[in_tie])+sum(branch.Pf[out_tie]) - sum(real(loss[r_[in_tie, out_tie]])) / 2,
                 sum(branch.Qt[in_tie])+sum(branch.Qf[out_tie]) - sum(imag(loss[r_[in_tie, out_tie]])) / 2  ))

        fd.write('\n----    ------  ------    --------    ------  ------    ------  ------')
        fd.write('\nTot:   %7.1f %7.1f    %7.1f    %7.2f %7.2f       -       -' %
            (-sum(bus.Vm[nzsh]**2 * bus.Gs[nzsh]),
             sum(bus.Vm[nzsh]**2 * bus.Bs[nzsh]),
             sum(fchg) + sum(tchg), sum(real(loss)), sum(imag(loss)) ))
        fd.write('\n')

    ## generator data
    if OUT_GEN:
        if isOPF:
            genlamP = bus.lam_p[e2i[gen.bus]]
            genlamQ = bus.lam_q[e2i[gen.bus]]

        fd.write('\n================================================================================')
        fd.write('\n|     Generator Data                                                           |')
        fd.write('\n================================================================================')
        fd.write('\n Gen   Bus   Status     Pg        Qg   ')
        if isOPF: fd.write('   Lambda ($/MVA-hr)')
        fd.write('\n  #     #              (MW)     (MVAr) ')
        if isOPF: fd.write('     P         Q    ')
        fd.write('\n----  -----  ------  --------  --------')
        if isOPF: fd.write('  --------  --------')
        for k in range(len(ong)):
            i = ong[k]
            fd.write('\n%3d %6d     %2d ' % (i, gen.bus[i], gen.status[i]))
            if (gen.status[i] > 0) & logical_or(gen.Pg[i], gen.Qg[i]):
                fd.write('%10.2f%10.2f' % (gen.Pg[i], gen.Qg[i]))
            else:
                fd.write('       -         -  ')
            if isOPF: fd.write('%10.2f%10.2f' % (genlamP[i], genlamQ[i]))

        fd.write('\n                     --------  --------')
        fd.write('\n            Total: %9.2f%10.2f' % (sum(gen.Pg[ong]), sum(gen.Qg[ong])))
        fd.write('\n')
        if any(onld + 1):
            fd.write('\n================================================================================')
            fd.write('\n|     Dispatchable Load Data                                                   |')
            fd.write('\n================================================================================')
            fd.write('\n Gen   Bus   Status     Pd        Qd   ')
            if isOPF: fd.write('   Lambda ($/MVA-hr)')
            fd.write('\n  #     #              (MW)     (MVAr) ')
            if isOPF: fd.write('     P         Q    ')
            fd.write('\n----  -----  ------  --------  --------')
            if isOPF: fd.write('  --------  --------')
            for k in range(len(onld)):
                i = onld[k]
                fd.write('\n%3d %6d     %2d ' % (i, gen.bus[i], gen.status[i]))
                if (gen.status[i] > 0) & logical_or(gen.Pg[i], gen.Qg[i]):
                    fd.write('%10.2f%10.2f' % (-gen.Pg[i], -gen.Qg[i]))
                else:
                    fd.write('       -         -  ')

                if isOPF: fd.write('%10.2f%10.2f' % (genlamP[i], genlamQ[i]))
            fd.write('\n                     --------  --------')
            fd.write('\n            Total: %9.2f%10.2f' % (-sum(gen.Pg[onld]), -sum(gen.Qg[onld])))
            fd.write('\n')

    ## bus data
    if OUT_BUS:
        fd.write('\n================================================================================')
        fd.write('\n|     Bus Data                                                                 |')
        fd.write('\n================================================================================')
        fd.write('\n Bus      Voltage          Generation             Load        ')
        if isOPF: fd.write('  Lambda($/MVA-hr)')
        fd.write('\n  #   Mag(pu) Ang(deg)   P (MW)   Q (MVAr)   P (MW)   Q (MVAr)')
        if isOPF: fd.write('     P        Q   ')
        fd.write('\n----- ------- --------  --------  --------  --------  --------')
        if isOPF: fd.write('  -------  -------')
        for i in range(nb):
            fd.write('\n%5d%7.3f%9.3f' % (i, bus.Vm[i], bus.Va[i]))
            if bus.bus_type[i] == REF:
                fd.write('*')
            else:
                fd.write(' ')
            g  = find((gen.status > 0) & (gen.bus == i) &
                        ~isload(gen))
            ld = find((gen.status > 0) & (gen.bus == i) &
                        isload(gen))
            if any(g + 1):
                fd.write('%9.2f%10.2f' % (sum(gen.Pg[g]), sum(gen.Qg[g])))
            else:
                fd.write('      -         -  ')

            if logical_or(bus.Pd[i], bus.Qd[i]) | any(ld + 1):
                if any(ld + 1):
                    fd.write('%10.2f*%9.2f*' % (bus.Pd[i] - sum(gen.Pg[ld]),
                                                bus.Qd[i] - sum(gen.Qg[ld])))
                else:
                    fd.write('%10.2f%10.2f ' % (bus.Pd[i], bus.Qd[i]))
            else:
                fd.write('       -         -   ')
            if isOPF:
                fd.write('%9.3f' % bus.lam_p[i])
                if abs(bus.lam_q[i]) > ptol:
                    fd.write('%8.3f' % bus.lam_q[i])
                else:
                    fd.write('     -')
        fd.write('\n                        --------  --------  --------  --------')
        fd.write('\n               Total: %9.2f %9.2f %9.2f %9.2f' %
            (sum(gen.Pg[ong]), sum(gen.Qg[ong]),
             sum(bus.Pd[nzld]) - sum(gen.Pg[onld]),
             sum(bus.Qd[nzld]) - sum(gen.Qg[onld])))
        fd.write('\n')

    ## branch data
    if OUT_BRANCH:
        fd.write('\n================================================================================')
        fd.write('\n|     Branch Data                                                              |')
        fd.write('\n================================================================================')
        fd.write('\nBrnch   From   To    From Bus Injection   To Bus Injection     Loss (I^2 * Z)  ')
        fd.write('\n  #     Bus    Bus    P (MW)   Q (MVAr)   P (MW)   Q (MVAr)   P (MW)   Q (MVAr)')
        fd.write('\n-----  -----  -----  --------  --------  --------  --------  --------  --------')
        for i in range(nl):
            fd.write('\n%4d%7d%7d%10.2f%10.2f%10.2f%10.2f%10.3f%10.2f' %
                (i, branch.f_bus[i], branch.t_bus[i],
                     branch.Pf[i], branch.Qf[i], branch.Pt[i], branch.Qt[i],
                     loss[i].real, loss[i].imag))
        fd.write('\n                                                             --------  --------')
        fd.write('\n                                                    Total:%10.3f%10.2f' %
                (sum(real(loss)), sum(imag(loss))))
        fd.write('\n')

    ##-----  constraint data  -----
    if isOPF:
        ctol = ppopt['OPF_VIOLATION']   ## constraint violation tolerance
        ## voltage constraints
        if (not isDC) & (OUT_V_LIM == 2 | (OUT_V_LIM == 1 &
                             (any(bus.Vm < bus.Vmin + ctol) |
                              any(bus.Vm > bus.Vmax - ctol) |
                              any(bus.mu_vmin > ptol) |
                              any(bus.mu_vmax > ptol)))):
            fd.write('\n================================================================================')
            fd.write('\n|     Voltage Constraints                                                      |')
            fd.write('\n================================================================================')
            fd.write('\nBus #  Vmin mu    Vmin    |V|   Vmax    Vmax mu')
            fd.write('\n-----  --------   -----  -----  -----   --------')
            for i in range(nb):
                if (OUT_V_LIM == 2) | (OUT_V_LIM == 1 &
                             ((bus.Vm[i] < bus.Vmin[i] + ctol) |
                              (bus.Vm[i] > bus.Vmax[i] - ctol) |
                              (bus.mu_vmin[i] > ptol) |
                              (bus.mu_vmax[i] > ptol))):
                    fd.write('\n%5d' % i)
                    if ((bus.Vm[i] < bus.Vmin[i] + ctol) |
                            (bus.mu_vmin[i] > ptol)):
                        fd.write('%10.3f' % bus.mu_vmin[i])
                    else:
                        fd.write('      -   ')

                    fd.write('%8.3f%7.3f%7.3f' % (i, bus.vmin, bus.Vm, bus.Vmax))
                    if (bus.Vm[i] > bus.Vmax[i] - ctol) | (bus.mu_vmax[i] > ptol):
                        fd.write('%10.3f' % bus.mu_vmax)
                    else:
                        fd.write('      -    ')
            fd.write('\n')

        ## generator P constraints
        if (OUT_PG_LIM == 2) | \
                ((OUT_PG_LIM == 1) & (any(gen.Pg[ong] < gen.Pmin[ong] + ctol) |
                                      any(gen.Pg[ong] > gen.Pmax[ong] - ctol) |
                                      any(gen.mu_pmin[ong] > ptol) |
                                      any(gen.mu_pmax[ong] > ptol))) | \
                ((not isDC) & ((OUT_QG_LIM == 2) |
                ((OUT_QG_LIM == 1) & (any(gen.Qg[ong] < gen.Qmin[ong] + ctol) |
                                      any(gen.Qg[ong] > gen.Qmax[ong] - ctol) |
                                      any(gen.mu_qmin[ong] > ptol) |
                                      any(gen.mu_qmax[ong] > ptol))))):
            fd.write('\n================================================================================')
            fd.write('\n|     Generation Constraints                                                   |')
            fd.write('\n================================================================================')

        if (OUT_PG_LIM == 2) | ((OUT_PG_LIM == 1) &
                                 (any(gen.Pg[ong] < gen.Pmin[ong] + ctol) |
                                  any(gen.Pg[ong] > gen.Pmax[ong] - ctol) |
                                  any(gen.mu_pmin[ong] > ptol) |
                                  any(gen.mu_pmax[ong] > ptol))):
            fd.write('\n Gen   Bus                Active Power Limits')
            fd.write('\n  #     #    Pmin mu    Pmin       Pg       Pmax    Pmax mu')
            fd.write('\n----  -----  -------  --------  --------  --------  -------')
            for k in range(len(ong)):
                i = ong[k]
                if (OUT_PG_LIM == 2) | ((OUT_PG_LIM == 1) &
                            ((gen.Pg[i] < gen.Pmin[i] + ctol) |
                             (gen.Pg[i] > gen.Pmax[i] - ctol) |
                             (gen.mu_pmin[i] > ptol) | (gen.mu_pmax[i] > ptol))):
                    fd.write('\n%4d%6d ' % (i, gen.bus[i]))
                    if (gen.Pg[i] < gen.Pmin[i] + ctol) | (gen.mu_pmin[i] > ptol):
                        fd.write('%8.3f' % gen.mu_pmin[i])
                    else:
                        fd.write('     -  ')
                    if gen.Pg[i]:
                        fd.write('%10.2f%10.2f%10.2f' % (i, gen.Pmin, gen.Pg, gen.Pmax))
                    else:
                        fd.write('%10.2f       -  %10.2f' % (i, gen.Pmin, gen.Pmax))
                    if (gen.Pg[i] > gen.Pmax[i] - ctol) | (gen.mu_pmax[i] > ptol):
                        fd.write('%9.3f' % gen.mu_pmax[i])
                    else:
                        fd.write('      -  ')
            fd.write('\n')

        ## generator Q constraints
        if (not isDC) & ((OUT_QG_LIM == 2) | ((OUT_QG_LIM == 1) &
                                 (any(gen.Qg[ong] < gen.Qmin[ong] + ctol) |
                                  any(gen.Qg[ong] > gen.Qmax[ong] - ctol) |
                                  any(gen.mu_qmin[ong] > ptol) |
                                  any(gen.mu_qmax[ong] > ptol)))):
            fd.write('\nGen  Bus              Reactive Power Limits')
            fd.write('\n #    #   Qmin mu    Qmin       Qg       Qmax    Qmax mu')
            fd.write('\n---  ---  -------  --------  --------  --------  -------')
            for k in range(len(ong)):
                i = ong[k]
                if (OUT_QG_LIM == 2) | ((OUT_QG_LIM == 1) &
                            ((gen.Qg[i] < gen.Qmin[i] + ctol) |
                             (gen.Qg[i] > gen.Qmax[i] - ctol) |
                             (gen.mu_qmin[i] > ptol) |
                             (gen.mu_qmax[i] > ptol))):
                    fd.write('\n%3d%5d' % (i, gen.bus[i]))
                    if (gen.Qg[i] < gen.Qmin[i] + ctol) | (gen.mu_qmin[i] > ptol):
                        fd.write('%8.3f' % gen.mu_qmin[i])
                    else:
                        fd.write('     -  ')
                    if gen.Qg[i]:
                        fd.write('%10.2f%10.2f%10.2f' % (gen.Qmin[i], gen.Qg[i], gen.Qmax[i]))
                    else:
                        fd.write('%10.2f       -  %10.2f' % (gen.Qmin[i], gen.Qmax[i]))

                    if (gen.Qg[i] > gen.Qmax[i] - ctol) | (gen.mu_qmax[i] > ptol):
                        fd.write('%9.3f' % gen.mu_qmax[i])
                    else:
                        fd.write('      -  ')
            fd.write('\n')

        ## dispatchable load P constraints
        if (OUT_PG_LIM == 2) | (OUT_QG_LIM == 2) | \
                ((OUT_PG_LIM == 1) & (any(gen.Pg[onld] < gen.Pmin[onld] + ctol) |
                                      any(gen.Pg[onld] > gen.Pmax[onld] - ctol) |
                                      any(gen.mu_pmin[onld] > ptol) |
                                      any(gen.mu_pmax[onld] > ptol))) | \
                ((OUT_QG_LIM == 1) & (any(gen.Qg[onld] < gen.Qmin[onld] + ctol) |
                                      any(gen.Qg[onld] > gen.Qmax[onld] - ctol) |
                                      any(gen.mu_qmin[onld] > ptol) |
                                      any(gen.mu_qmax[onld] > ptol))):
            fd.write('\n================================================================================')
            fd.write('\n|     Dispatchable Load Constraints                                            |')
            fd.write('\n================================================================================')
        if (OUT_PG_LIM == 2) | ((OUT_PG_LIM == 1) &
                                 (any(gen.Pg[onld] < gen.Pmin[onld] + ctol) |
                                  any(gen.Pg[onld] > gen.Pmax[onld] - ctol) |
                                  any(gen.mu_pmin[onld] > ptol) |
                                  any(gen.mu_pmax[onld] > ptol))):
            fd.write('\nGen  Bus               Active Power Limits')
            fd.write('\n #    #   Pmin mu    Pmin       Pg       Pmax    Pmax mu')
            fd.write('\n---  ---  -------  --------  --------  --------  -------')
            for k in range(len(onld)):
                i = onld[k]
                if (OUT_PG_LIM == 2) | ((OUT_PG_LIM == 1) &
                            ((gen.Pg[i] < gen.Pmin[i] + ctol) |
                             (gen.Pg[i] > gen.Pmax[i] - ctol) |
                             (gen.mu_pmin[i] > ptol) |
                             (gen.mu_pmax[i] > ptol))):
                    fd.write('\n%3d%5d' % (i, gen.bus[i]))
                    if (gen.Pg[i] < gen.Pmin[i] + ctol) | (gen.mu_pmin[i] > ptol):
                        fd.write('%8.3f' % gen.mu_pmin[i])
                    else:
                        fd.write('     -  ')
                    if gen.Pg[i]:
                        fd.write('%10.2f%10.2f%10.2f' % (gen.Pmin[i], gen.Pg[i], gen.Pmax[i]))
                    else:
                        fd.write('%10.2f       -  %10.2f' % (gen.Pmin[i], gen.Pmax[i]))

                    if (gen.pg[i] > gen.Pmax[i] - ctol) | (gen.mu_pmax[i] > ptol):
                        fd.write('%9.3f' % gen.mu_pmax[i])
                    else:
                        fd.write('      -  ')
            fd.write('\n')

        ## dispatchable load Q constraints
        if (not isDC) & ((OUT_QG_LIM == 2) | ((OUT_QG_LIM == 1) &
                                 (any(gen.Qg[onld] < gen.Qmin[onld] + ctol) |
                                  any(gen.Qg[onld] > gen.Qmax[onld] - ctol) |
                                  any(gen.mu_qmin[onld] > ptol) |
                                  any(gen.mu_qmax[onld] > ptol)))):
            fd.write('\nGen  Bus              Reactive Power Limits')
            fd.write('\n #    #   Qmin mu    Qmin       Qg       Qmax    Qmax mu')
            fd.write('\n---  ---  -------  --------  --------  --------  -------')
            for k in range(len(onld)):
                i = onld[k]
                if (OUT_QG_LIM == 2) | ((OUT_QG_LIM == 1) &
                            ((gen.Qg[i] < gen.Qmin[i] + ctol) |
                             (gen.Qg[i] > gen.Qmax[i] - ctol) |
                             (gen.mu_qmin[i] > ptol) |
                             (gen.mu_qmax[i] > ptol))):
                    fd.write('\n%3d%5d' % (i, gen.bus[i]))
                    if (gen.Qg[i] < gen.Qmin[i] + ctol) | (gen.mu_qmin[i] > ptol):
                        fd.write('%8.3f' % gen.mu_qmin[i])
                    else:
                        fd.write('     -  ')

                    if gen.Qg[i]:
                        fd.write('%10.2f%10.2f%10.2f' % (gen.Qmin[i], gen.Qg[i], gen.Qmax[i]))
                    else:
                        fd.write('%10.2f       -  %10.2f' % (gen.Qmin[i], gen.Qmax[i]))

                    if (gen.Qg[i] > gen.Qmax[i] - ctol) | (gen.mu_qmax[i] > ptol):
                        fd.write('%9.3f' % gen.mu_qmax[i])
                    else:
                        fd.write('      -  ')
            fd.write('\n')

        ## line flow constraints
        if (ppopt['OPF_FLOW_LIM'] == 1) | isDC:  ## P limit
            Ff = branch.Pf
            Ft = branch.Pt
            strg = '\n  #     Bus    Pf  mu     Pf      |Pmax|      Pt      Pt  mu   Bus'
        elif ppopt['OPF_FLOW_LIM'] == 2:   ## |I| limit
            Ff = abs( (branch.Pf + 1j * branch.Qf) / V[e2i[branch.f_bus]] )
            Ft = abs( (branch.Pt + 1j * branch.Qt) / V[e2i[branch.t_bus]] )
            strg = '\n  #     Bus   |If| mu    |If|     |Imax|     |It|    |It| mu   Bus'
        else:                ## |S| limit
            Ff = abs(branch.Pf + 1j * branch.Qf)
            Ft = abs(branch.Pt + 1j * branch.Qt)
            strg = '\n  #     Bus   |Sf| mu    |Sf|     |Smax|     |St|    |St| mu   Bus'

        if (OUT_LINE_LIM == 2) | ((OUT_LINE_LIM == 1) &
                            (any((branch.rate_a != 0) & (abs(Ff) > branch.rate_a - ctol)) |
                             any((branch.rate_a != 0) & (abs(Ft) > branch.rate_a - ctol)) |
                             any(branch.mu_Sf > ptol) |
                             any(branch.mu_St > ptol))):
            fd.write('\n================================================================================')
            fd.write('\n|     Branch Flow Constraints                                                  |')
            fd.write('\n================================================================================')
            fd.write('\nBrnch   From     "From" End        Limit       "To" End        To')
            fd.write(strg)
            fd.write('\n-----  -----  -------  --------  --------  --------  -------  -----')
            for i in range(nl):
                if (OUT_LINE_LIM == 2) | ((OUT_LINE_LIM == 1) &
                       (((branch.rate_a[i] != 0) & (abs(Ff[i]) > branch.rate_a[i] - ctol)) |
                        ((branch.rate_a[i] != 0) & (abs(Ft[i]) > branch.rate_a[i] - ctol)) |
                        (branch.mu_Sf[i] > ptol) | (branch.mu_St > ptol))):
                    fd.write('\n%4d%7d' % (i, branch.f_bus[i]))
                    if (Ff[i] > branch.rate_a[i] - ctol) | (branch.mu_Sf[i] > ptol):
                        fd.write('%10.3f' % branch.mu_Sf[i])
                    else:
                        fd.write('      -   ')

                    fd.write('%9.2f%10.2f%10.2f' %
                        (Ff[i], branch.rate_a[i], Ft[i]))
                    if (Ft[i] > branch.rate_a[i] - ctol) | (branch.mu_St[i] > ptol):
                        fd.write('%10.3f' % branch.mu_St[i])
                    else:
                        fd.write('      -   ')
                    fd.write('%6d' % branch.t_bus[i])
            fd.write('\n')

    ## execute userfcn callbacks for 'printpf' stage
#     if have_results_struct and 'userfcn' in results:
#         if not isOPF:  ## turn off option for all constraints if it isn't an OPF
#             ppopt = ppoption(ppopt, 'OUT_ALL_LIM', 0)
#         run_userfcn(results["userfcn"], 'printpf', results, fd, ppopt)
