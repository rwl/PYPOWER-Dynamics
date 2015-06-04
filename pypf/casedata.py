from numpy import ones, zeros

from pypower.idx_bus import \
    BUS_TYPE, PD, QD, GS, BS, VM, VA, BASE_KV , BUS_AREA
    
from pypower.idx_gen import \
    GEN_BUS, PG, QG, QMAX, QMIN, VG, GEN_STATUS, PMAX, PMIN
    
from pypower.idx_brch import \
    F_BUS, T_BUS, BR_R, BR_X, BR_B, TAP, SHIFT, BR_STATUS
    
from pypower.ext2int import ext2int

class PFCase(object):
    def __init__(self, nb = 0, ng = 0, nl = 0):
        self.baseMVA = 100
        self.bus = Bus(nb)
        self.gen = Gen(ng)
        self.branch = Branch(nl)

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
    
    _gen = ppc['gen']
    gen = Gen()
    gen.n = _gen.shape[0]
    gen.bus = _gen[:, GEN_BUS].astype(int)
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
    br.f_bus = _br[:, F_BUS].astype(int)
    br.t_bus = _br[:, T_BUS].astype(int)
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
