#!python3
#
# Copyright (C) 2014-2015 Julius Susanto. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

"""
PYPOWER-Dynamics
Build modified Ybus matrix

"""

import numpy as np

def mod_Ybus(Ybus, elements, bus, gen, baseMVA):
    # Add equivalent generator and grid admittances to Ybus matrix
    for element in elements.values():
        Ye = 0
        
        # 4th/6th order machines and converters
        if element.__module__ in ['pydyn.sym_order4', 'pydyn.sym_order6a', 'pydyn.sym_order6b', 'pydyn.vsc_average']:
            i = gen.bus[element.gen_no]
            Ye = element.Yg
        
        # External grid
        if element.__module__ == 'pydyn.ext_grid':
            i = gen.bus[element.gen_no]
            Ye = 1 / (1j * element.params['Xdp'])
        
        if Ye != 0:
            Ybus[i,i] = Ybus[i,i] + Ye

    # Add equivalent load admittance to Ybus matrix    
    Pl, Ql = bus.Pd, bus.Qd
    for i in range(len(Pl)):
        S_load = (Pl[i] - 1j * Ql[i]) / baseMVA
        y_load = S_load / (bus.Vm[i] ** 2)
        Ybus[i,i] = Ybus[i,i] + y_load
    
    return Ybus
