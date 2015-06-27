from numpy import ones, zeros, copy, delete, array

from pypower.idx_bus import \
    BUS_I, BUS_TYPE, PD, QD, GS, BS, VM, VA, BASE_KV , BUS_AREA
    
from pypower.idx_gen import \
    GEN_BUS, PG, QG, QMAX, QMIN, VG, GEN_STATUS, PMAX, PMIN
    
from pypower.idx_brch import \
    F_BUS, T_BUS, BR_R, BR_X, BR_B, TAP, SHIFT, BR_STATUS
    
from pypower.ext2int import ext2int

# NR = 1 # Newton's method
# FDXB = 2 # Fast-Decoupled (XB version)
# FDBX = 3 # Fast-Decoupled (BX version)

class PFCase(object):
    def __init__(self, nb = 0, ng = 0, nl = 0):
        self.baseMVA = 100
        self.bus = Bus(nb)
        self.gen = Gen(ng)
        self.branch = Branch(nl)

    def copy(self):
        pfc = PFCase()
        pfc.baseMVA = self.baseMVA
        pfc.bus = self.bus.copy()
        pfc.gen = self.gen.copy()
        pfc.branch = self.branch.copy()
        return pfc

class Bus(object):
    def __init__(self, n = 0):
        self.n = n
        self.bus_type = ones(n)
        self.Pd = zeros(n)
        self.Qd = zeros(n)
        self.Gs = zeros(n)
        self.Bs = zeros(n)
        self.Vm = zeros(n)
        self.Va = zeros(n)
        self.baseKV = zeros(n)

        self.area = zeros(n)

    def copy(self):
        bus = Bus()
        bus.n = self.n
        bus.bus_type = copy(self.bus_type)
        bus.Pd = copy(self.Pd)
        bus.Qd = copy(self.Qd)
        bus.Gs = copy(self.Gs)
        bus.Bs = copy(self.Bs)
        bus.Vm = copy(self.Vm)
        bus.Va = copy(self.Va)
        bus.baseKV = copy(self.baseKV)

        bus.area = copy(self.area)
        return bus

class Gen(object):
    def __init__(self, n = 0):
        self.n = n;
        self.bus = zeros(n)
        self.Pg = zeros(n)
        self.Qg = zeros(n)
        self.Qmax = zeros(n)
        self.Qmin = zeros(n)
        self.Vg = zeros(n)
        self.status = ones(n)

        self.Pmin = zeros(n)
        self.Pmax = zeros(n)

    def copy(self):
        gen = Gen()
        gen.n = self.n;
        gen.bus = copy(self.bus)
        gen.Pg = copy(self.Pg)
        gen.Qg = copy(self.Qg)
        gen.Qmax = copy(self.Qmax)
        gen.Qmin = copy(self.Qmin)
        gen.Vg = copy(self.Vg)
        gen.status = copy(self.status)

        gen.Pmin = copy(self.Pmin)
        gen.Pmax = copy(self.Pmax)
        return gen

class Branch(object):
    def __init__(self, n = 0):
        self.n = n
        self.f_bus = zeros(n)
        self.t_bus = zeros(n)
        self.r = zeros(n)
        self.x = zeros(n)
        self.b = zeros(n)
        self.tap = zeros(n)
        self.shift = zeros(n)
        self.status = zeros(n)
        self.Pf = zeros(n)
        self.Qf = zeros(n)
        self.Pt = zeros(n)
        self.Qt = zeros(n)

    def copy(self):
        branch = Branch()
        branch.n = self.n
        branch.f_bus = copy(self.f_bus)
        branch.t_bus = copy(self.t_bus)
        branch.r = copy(self.r)
        branch.x = copy(self.x)
        branch.b = copy(self.b)
        branch.tap = copy(self.tap)
        branch.shift = copy(self.shift)
        branch.status = copy(self.status)
        branch.Pf = copy(self.Pf)
        branch.Qf = copy(self.Qf)
        branch.Pt = copy(self.Pt)
        branch.Qt = copy(self.Qt)
        return branch

    def delete(self, i):
        self.n -= 1
        self.f_bus = delete(self.f_bus, i)
        self.t_bus = delete(self.t_bus, i)
        self.r = delete(self.r, i)
        self.x = delete(self.x, i)
        self.b = delete(self.b, i)
        self.tap = delete(self.tap, i)
        self.shift = delete(self.shift, i)
        self.status = delete(self.status, i)
        if (len(self.Pf) != 0):
            self.Pf = delete(self.Pf, i)
            self.Qf = delete(self.Qf, i)
            self.Pt = delete(self.Pt, i)
            self.Qt = delete(self.Qt, i)

# class PFOptions(object):
#     def __init__(self):
#         # power flow algorithm
#         self.pf_alg = NR
#         # termination tolerance on per unit P & Q mismatch
#         self.pf_tol = 1e-8
#         # maximum number of iterations for Newton's method
#         self.pf_max_it = 10
#         # maximum number of iterations for fast decoupled method
#         self.pf_max_it_fd = 30
#         # enforce gen reactive power limits, at expense of |V|
#         self.enforce_q_lims = False
#         # use DC power flow formulation
#         self.pf_dc = False
#         
#         ## Output options
#         # amount of progress info printed
#         self.verbose = 1
#         # controls printing of results:
#         #   -1 - individual flags control what prints,
#         #    0 - don't print anything (overrides individual flags),
#         #    1 - print everything (overrides individual flags)
#         self.out_all = -1
#         # print system summary
#         self.out_sys_sum = True
#         # print area summaries
#         self.out_area_sum = False
#         # print bus detail
#         self.out_bus = True
#         # print branch detail
#         self.out_branch = True
#         # print generator detail
#         self.out_gen = False
        
def convert_ppc(ppc):
    ppc = ext2int(ppc)

    pfc = PFCase()
    pfc.baseMVA = ppc["baseMVA"]

    _bus = ppc['bus']    
    bus = Bus()
    bus.n = _bus.shape[0]
    bus.bus_type = _bus[:, BUS_TYPE].copy()
    bus.Pd = _bus[:, PD].copy()
    bus.Qd = _bus[:, QD].copy()
    bus.Gs = _bus[:, GS].copy()
    bus.Bs = _bus[:, BS].copy()
    bus.Vm = _bus[:, VM].copy()
    bus.Va = _bus[:, VA].copy()
    bus.baseKV = _bus[:, BASE_KV].copy()

    bus.area = _bus[:, BUS_AREA].copy()

    bmap = {bi: i for i, bi in enumerate(_bus[:, BUS_I].astype(int))}

    _gen = ppc['gen']
    gen = Gen()
    gen.n = _gen.shape[0]
    gbus = [bmap[i] for i in _gen[:, GEN_BUS].astype(int)]
    gen.bus = array(gbus)
    gen.Pg = _gen[:, PG].copy()
    gen.Qg = _gen[:, QG].copy()
    gen.Qmax = _gen[:, QMAX].copy()
    gen.Qmin = _gen[:, QMIN].copy()
    gen.Vg = _gen[:, VG].copy()
    gen.status = _gen[:, GEN_STATUS].astype(int)

    gen.Pmin = _gen[:, PMIN].copy()
    gen.Pmax = _gen[:, PMAX].copy()

    _br = ppc['branch']
    br = Branch()
    br.n = _br.shape[0]
    fbus = [bmap[i] for i in _br[:, F_BUS].astype(int)]
    tbus = [bmap[i] for i in _br[:, T_BUS].astype(int)]
    br.f_bus = array(fbus)
    br.t_bus = array(tbus)
    br.r = _br[:, BR_R].copy()
    br.x = _br[:, BR_X].copy()
    br.b = _br[:, BR_B].copy()
    br.tap = _br[:, TAP].copy()
    br.shift = _br[:, SHIFT].copy()
    br.status = _br[:, BR_STATUS].astype(int)

    pfc.bus = bus
    pfc.gen = gen
    pfc.branch = br

    return pfc
