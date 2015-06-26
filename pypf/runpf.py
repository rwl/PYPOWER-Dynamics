# Copyright (c) 1996-2015 PSERC. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

"""Runs a power flow.
"""

from sys import stdout, stderr

from os.path import dirname, join

from time import time

from numpy import r_, c_, ix_, zeros, pi, ones, exp, argmax, array
from numpy import flatnonzero as find

# from pypower.ext2int import ext2int
# from pypower.loadcase import loadcase
from pypower.ppoption import ppoption
from pypower.ppver import ppver
from pypower.dcpf import dcpf
from pypower.savecase import savecase
from pypower.int2ext import int2ext
from pypower.idx_bus import REF, PQ

from pypf.bustypes import bustypes
from pypf.makeBdc import makeBdc
from pypf.makeSbus import makeSbus
from pypf.newtonpf import newtonpf
from pypf.fdpf import fdpf
from pypf.makeB import makeB
from pypf.pfsoln import pfsoln
from pypf.printpf import printpf

from pypf.makeYbus import makeYbus


def runpf(casedata=None, ppopt=None, fname='', solvedcase=''):
    """Runs a power flow.

    Runs a power flow [full AC Newton's method by default] and optionally
    returns the solved values in the data matrices, a flag which is C{True} if
    the algorithm was successful in finding a solution, and the elapsed
    time in seconds. All input arguments are optional. If C{casename} is
    provided it specifies the name of the input data file or dict
    containing the power flow data. The default value is 'case9'.

    If the ppopt is provided it overrides the default PYPOWER options
    vector and can be used to specify the solution algorithm and output
    options among other things. If the 3rd argument is given the pretty
    printed output will be appended to the file whose name is given in
    C{fname}. If C{solvedcase} is specified the solved case will be written
    to a case file in PYPOWER format with the specified name. If C{solvedcase}
    ends with '.mat' it saves the case as a MAT-file otherwise it saves it
    as a Python-file.

    If the C{ENFORCE_Q_LIMS} options is set to C{True} [default is false] then
    if any generator reactive power limit is violated after running the AC
    power flow, the corresponding bus is converted to a PQ bus, with Qg at
    the limit, and the case is re-run. The voltage magnitude at the bus
    will deviate from the specified value in order to satisfy the reactive
    power limit. If the reference bus is converted to PQ, the first
    remaining PV bus will be used as the slack bus for the next iteration.
    This may result in the real power output at this generator being
    slightly off from the specified values.

    Enforcing of generator Q limits inspired by contributions from Mu Lin,
    Lincoln University, New Zealand (1/14/05).

    @author: Ray Zimmerman (PSERC Cornell)
    """
    ## default arguments
    if casedata is None:
        casedata = join(dirname(__file__), 'case9')
    ppopt = ppoption(ppopt)

    ## options
    verbose = ppopt["VERBOSE"]
    qlim = ppopt["ENFORCE_Q_LIMS"]  ## enforce Q limits on gens?
    dc = ppopt["PF_DC"]             ## use DC formulation?

    ## read data
#     ppc = loadcase(casedata)
    ppc = casedata.copy()

    ## add zero columns to branch for flows if needed
    nl = ppc.branch.n
    if len(ppc.branch.Pf) == 0:
        ppc.branch.Pf = zeros(nl)
    if len(ppc.branch.Qf) == 0:
        ppc.branch.Qf = zeros(nl)
    if len(ppc.branch.Pt) == 0:
        ppc.branch.Pt = zeros(nl)
    if len(ppc.branch.Qt) == 0:
        ppc.branch.Qt = zeros(nl)

    ## convert to internal indexing
#     ppc = ext2int(ppc)
    baseMVA, bus, gen, branch = ppc.baseMVA, ppc.bus, ppc.gen, ppc.branch

    ## get bus index lists of each type of bus
    ref, pv, pq = bustypes(bus, gen)

    ## generator info
    on = find(array(gen.status) > 0)      ## which generators are on?
    gbus = gen.bus[on].astype(int)    ## what buses are they at?

    ##-----  run the power flow  -----
    t0 = time()
    if verbose > 0:
        v = ppver('all')
        stdout.write('PYPOWER Version %s, %s' % (v["Version"], v["Date"]))

    if dc:                               # DC formulation
        if verbose:
            stdout.write(' -- DC Power Flow\n')

        ## initial state
        Va0 = bus.Va * (pi / 180)

        ## build B matrices and phase shift injections
        B, Bf, Pbusinj, Pfinj = makeBdc(baseMVA, bus, branch)

        ## compute complex bus power injections [generation - load]
        ## adjusted for phase shifters and real shunts
        Pbus = makeSbus(baseMVA, bus, gen).real - Pbusinj - bus.Gs / baseMVA

        ## "run" the power flow
        Va = dcpf(B, Pbus, Va0, ref, pv, pq)

        ## update data matrices with solution
        branch.Qf = zeros(branch.n)
        branch.Qt = zeros(branch.n)
        branch.Pf = (Bf * Va + Pfinj) * baseMVA
        branch.Pt = -array(branch.Pf)
        bus.Vm = ones(bus.n)
        bus.Va = Va * (180 / pi)
        ## update Pg for slack generator (1st gen at ref bus)
        ## (note: other gens at ref bus are accounted for in Pbus)
        ##      Pg = Pinj + Pload + Gs
        ##      newPg = oldPg + newPinj - oldPinj
        refgen = zeros(len(ref), dtype=int)
        for k in range(len(ref)):
            temp = find(gbus == ref[k])
            refgen[k] = on[temp[0]]
        gen.Pg[refgen] = gen.Pg[refgen] + (B[ref, :] * Va - Pbus[ref]) * baseMVA

        success = 1
    else:                                ## AC formulation
        alg = ppopt['PF_ALG']
        if verbose > 0:
            if alg == 1:
                solver = 'Newton'
            elif alg == 2:
                solver = 'fast-decoupled, XB'
            elif alg == 3:
                solver = 'fast-decoupled, BX'
            elif alg == 4:
                solver = 'Gauss-Seidel'
            else:
                solver = 'unknown'
            print(' -- AC Power Flow (%s)\n' % solver)

        ## initial state
        # V0    = ones(bus.shape[0])            ## flat start
        V0  = bus.Vm * exp(1j * pi/180 * bus.Va)
        V0[gbus] = gen.Vg[on] / abs(V0[gbus]) * V0[gbus]

        if qlim:
            ref0 = ref                         ## save index and angle of
            Varef0 = bus.Va[ref0]             ##   original reference bus(es)
            limited = []                       ## list of indices of gens @ Q lims
            fixedQg = zeros(gen.shape[0])      ## Qg of gens at Q limits

        repeat = True
        while repeat:
            ## build admittance matrices
            Ybus, Yf, Yt = makeYbus(baseMVA, bus, branch)

            ## compute complex bus power injections [generation - load]
            Sbus = makeSbus(baseMVA, bus, gen)

            ## run the power flow
            alg = ppopt["PF_ALG"]
            if alg == 1:
                V, success, _ = newtonpf(Ybus, Sbus, V0, ref, pv, pq, ppopt)
            elif alg == 2 or alg == 3:
                Bp, Bpp = makeB(baseMVA, bus, branch, alg)
                V, success, _ = fdpf(Ybus, Sbus, V0, Bp, Bpp, ref, pv, pq, ppopt)
#             elif alg == 4:
#                 V, success, _ = gausspf(Ybus, Sbus, V0, ref, pv, pq, ppopt)
            else:
                stderr.write('Only Newton''s method and fast-decoupled '
                             'power flow algorithms currently '
                             'implemented.\n')

            ## update data matrices with solution
            bus, gen, branch = pfsoln(baseMVA, bus, gen, branch, Ybus, Yf, Yt, V, ref, pv, pq)

            if qlim:             ## enforce generator Q limits
                ## find gens with violated Q constraints
                gen_status = gen.status > 0
                qg_max_lim = gen.Qg > gen.Qmax
                qg_min_lim = gen.Qg < gen.Qmin
                
                mx = find( gen_status & qg_max_lim )
                mn = find( gen_status & qg_min_lim )
                
                if len(mx) > 0 or len(mn) > 0:  ## we have some Q limit violations
                    # No PV generators
                    if len(pv) == 0:
                        if verbose:
                            if len(mx) > 0:
                                print('Gen %d [only one left] exceeds upper Q limit : INFEASIBLE PROBLEM\n' % mx + 1)
                            else:
                                print('Gen %d [only one left] exceeds lower Q limit : INFEASIBLE PROBLEM\n' % mn + 1)

                        success = 0
                        break

                    ## one at a time?
                    if qlim == 2:    ## fix largest violation, ignore the rest
                        k = argmax(r_[gen.Qg[mx] - gen.Qmax[mx],
                                      gen.Qmin[mn] - gen.Qg[mn]])
                        if k > len(mx):
                            mn = mn[k - len(mx)]
                            mx = []
                        else:
                            mx = mx[k]
                            mn = []

                    if verbose and len(mx) > 0:
                        for i in range(len(mx)):
                            print('Gen ' + str(mx[i] + 1) + ' at upper Q limit, converting to PQ bus\n')

                    if verbose and len(mn) > 0:
                        for i in range(len(mn)):
                            print('Gen ' + str(mn[i] + 1) + ' at lower Q limit, converting to PQ bus\n')

                    ## save corresponding limit values
                    fixedQg[mx] = gen.Qmax[mx]
                    fixedQg[mn] = gen.Qmin[mn]
                    mx = r_[mx, mn].astype(int)

                    ## convert to PQ bus
                    gen.Qg[mx] = fixedQg[mx]      ## set Qg to binding 
                    for i in range(len(mx)):            ## [one at a time, since they may be at same bus]
                        gen.status[mx[i]] = 0        ## temporarily turn off gen,
                        bi = gen.bus[mx[i]]   ## adjust load accordingly,
                        bus.Pd[bi] = bus.Pd[bi] - gen.Pg[mx[i]]
                        bus.Qd[bi] = bus.Qd[bi] - gen.Qg[mx[i]]
                    
                    if len(ref) > 1 and any(bus.type[gen.bus[mx]] == REF):
                        raise ValueError('Sorry, PYPOWER cannot enforce Q '
                                         'limits for slack buses in systems '
                                         'with multiple slacks.')
                    
                    bus.type[gen.bus[mx].astype(int)] = PQ   ## & set bus type to PQ

                    ## update bus index lists of each type of bus
                    ref_temp = ref
                    ref, pv, pq = bustypes(bus, gen)
                    if verbose and ref != ref_temp:
                        print('Bus %d is new slack bus\n' % ref)

                    limited = r_[limited, mx].astype(int)
                else:
                    repeat = 0 ## no more generator Q limits violated
            else:
                repeat = 0     ## don't enforce generator Q limits, once is enough

        if qlim and len(limited) > 0:
            ## restore injections from limited gens [those at Q limits]
            gen.Qg[limited] = fixedQg[limited]    ## restore Qg value,
            for i in range(len(limited)):               ## [one at a time, since they may be at same bus]
                bi = gen.bus[limited[i]]           ## re-adjust load,
                bus.Pd[bi] = bus.Pd[bi] - gen.Pg[limited[i]]
                bus.Qd[bi] = bus.Qd[bi] - gen.Qg[limited[i]]
                gen.status[limited[i]] = 1           ## and turn gen back on
            
            if ref != ref0:
                ## adjust voltage angles to make original ref bus correct
                bus.Va = bus.Va - bus.Va[ref0] + Varef0

    ppc.et = time() - t0
    ppc.success = success

    ##-----  output results  -----
    ## convert back to original bus numbering & print results
    ppc.bus, ppc.gen, ppc.branch = bus, gen, branch
    results = ppc#int2ext(ppc)

    ## zero out result fields of out-of-service gens & branches
#     if len(results["order"]["gen"]["status"]["off"]) > 0:
#         results.gen.Pg[results["order"]["gen"]["status"]["off"]] = 0
#         results.gen.Qg[results["order"]["gen"]["status"]["off"]] = 0
# 
#     if len(results["order"]["branch"]["status"]["off"]) > 0:
#         results.branch.Pf[results["order"]["branch"]["status"]["off"]] = 0
#         results.branch.Qf[results["order"]["branch"]["status"]["off"]] = 0
#         results.branch.Pt[results["order"]["branch"]["status"]["off"]] = 0
#         results.branch.Qt[results["order"]["branch"]["status"]["off"]] = 0

    if fname:
        fd = None
        try:
            fd = open(fname, "a")
        except Exception as detail:
            stderr.write("Error opening %s: %s.\n" % (fname, detail))
        finally:
            if fd is not None:
                printpf(results, fd, ppopt)
                fd.close()
    else:
        printpf(results, stdout, ppopt)

    ## save solved case
    if solvedcase:
        savecase(solvedcase, results)

    return results, success


if __name__ == '__main__':
    runpf()
