#!python3
#
# Copyright (C) 2014-2015 Julius Susanto. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

"""
PYPOWER-Dynamics
Time-domain simulation engine

"""

from pydyn.interface import init_interfaces
from pydyn.mod_Ybus import mod_Ybus
from pydyn.version import pydyn_ver

from scipy.sparse.linalg import splu
import numpy as np

from pypf.runpf import runpf
from pypf.makeYbus import makeYbus
    
def run_sim(ppc, elements, dynopt = None, events = None, recorder = None):
    """
    Run a time-domain simulation
    
    Inputs:
        ppc         PYPOWER load flow case
        elements    Dictionary of dynamic model objects (machines, controllers, etc) with Object ID as key
        events      Events object
        recorder    Recorder object (empty)
    
    Outputs:
        recorder    Recorder object (with data)
    """
    
    #########
    # SETUP #
    #########
    
    # Get version information
    ver = pydyn_ver()
    print('PYPOWER-Dynamics ' + ver['Version'] + ', ' + ver['Date'])
    
    # Program options
    if dynopt:
        h = dynopt['h']             
        t_sim = dynopt['t_sim']           
        max_err = dynopt['max_err']        
        max_iter = dynopt['max_iter']
        verbose = dynopt['verbose']
    else:
        # Default program options
        h = 0.01                # step length (s)
        t_sim = 5               # simulation time (s)
        max_err = 0.0001        # Maximum error in network iteration (voltage mismatches)
        max_iter = 25           # Maximum number of network iterations
        verbose = False
        
    # Make lists of current injection sources (generators, external grids, etc) and controllers
    sources = []
    controllers = []
    for element in elements.values():
        if element.__module__ in ['pydyn.sym_order6a', 'pydyn.sym_order6b', 'pydyn.sym_order4', 'pydyn.ext_grid', 'pydyn.vsc_average', 'pydyn.asym_1cage', 'pydyn.asym_2cage']:
            sources.append(element)
            
        if element.__module__ == 'pydyn.controller':
            controllers.append(element)
    
    # Set up interfaces
    interfaces = init_interfaces(elements)
    
    ##################
    # INITIALISATION #
    ##################
    print('Initialising models...')
    
    # Run power flow and update bus voltages and angles in PYPOWER case object
    results, success = runpf(ppc) 
    ppc.bus.Vm = results.bus.Vm
    ppc.bus.Va = results.bus.Va
    
    # Build Ybus matrix
    ppc_int = ppc.copy()#ext2int(ppc)
    baseMVA, bus, branch = ppc_int.baseMVA, ppc_int.bus, ppc_int.branch
    Ybus, Yf, Yt = makeYbus(baseMVA, bus, branch)
    
    # Build modified Ybus matrix
    Ybus = mod_Ybus(Ybus, elements, bus, ppc_int.gen, baseMVA)
    
    # Calculate initial voltage phasors
    v0 = bus.Vm * (np.cos(np.radians(bus.Va)) + 1j * np.sin(np.radians(bus.Va)))
#     v0  = bus.Vm * np.exp(1j * np.pi/180 * bus.Va)
    
    # Initialise sources from load flow
    for source in sources:
        if source.__module__ in ['pydyn.asym_1cage', 'pydyn.asym_2cage']:
            # Asynchronous machine
            source_bus = source.bus_no#ppc_int['bus'][source.bus_no,0]
            v_source = v0[source_bus]
            source.initialise(v_source,0)
        else:
            # Generator or VSC
            source_bus = ppc_int.gen.bus[source.gen_no]
            S_source = np.complex(results.gen.Pg[source.gen_no] / baseMVA, results.gen.Qg[source.gen_no] / baseMVA)
            v_source = v0[source_bus]
            source.initialise(v_source,S_source)
    
    # Interface controllers and machines (for initialisation)
    for intf in interfaces:
        int_type = intf[0]
        var_name = intf[1]
        if int_type == 'OUTPUT':
            # If an output, interface in the reverse direction for initialisation
            intf[2].signals[var_name] = intf[3].signals[var_name]
        else:
            # Inputs are interfaced in normal direction during initialisation
            intf[3].signals[var_name] = intf[2].signals[var_name]
    
    # Initialise controllers
    for controller in controllers:
        controller.initialise()
    
    #############
    # MAIN LOOP #
    #############
    
    if events == None:
        print('Warning: no events!')
    
    # Factorise Ybus matrix
    Ybus_inv = splu(Ybus)
    
    y1 = []
    v_prev = v0
    print('Simulating...')
    for t in range(int(t_sim / h) + 1):
        if np.mod(t,1/h) == 0:
            print('t=' + str(t*h) + 's')
            
        # Interface controllers and machines
        for intf in interfaces:
            var_name = intf[1]
            intf[3].signals[var_name] = intf[2].signals[var_name]
        
        # Solve differential equations
        for j in range(4):
            # Solve step of differential equations
            for element in elements.values():
                element.solve_step(h,j) 
            
            # Interface with network equations
            v_prev = solve_network(sources, v_prev, Ybus_inv, ppc_int, bus.n, max_err, max_iter)
        
        if recorder != None:
            # Record signals or states
            recorder.record_variables(t*h, elements)
        
        if events != None:
            # Check event stack
            ppc, refactorise = events.handle_events(np.round(t*h,5), elements, ppc, baseMVA)
            
            if refactorise == True:
                # Rebuild Ybus from new ppc_int
                ppc_int = ppc.copy()#ext2int(ppc)
                baseMVA, bus, branch = ppc_int.baseMVA, ppc_int.bus, ppc_int.branch
                Ybus, Yf, Yt = makeYbus(baseMVA, bus, branch)
                
                # Rebuild modified Ybus
                Ybus = mod_Ybus(Ybus, elements, bus, ppc_int.gen, baseMVA)
                
                # Refactorise Ybus
                Ybus_inv = splu(Ybus)
                
                # Solve network equations
                v_prev = solve_network(sources, v_prev, Ybus_inv, ppc_int, bus.n, max_err, max_iter)
                
    return recorder
    
def solve_network(sources, v_prev, Ybus_inv, ppc_int, no_buses, max_err, max_iter):
    """
    Solve network equations
    """
    verr = 1
    i = 1
    # Iterate until network voltages in successive iterations are within tolerance
    while verr > max_err and i < max_iter:        
        # Update current injections for sources
        I = np.zeros(no_buses, dtype='complex')
        for source in sources:
            if source.__module__ in ['pydyn.asym_1cage', 'pydyn.asym_2cage']:
                # Asynchronous machine
                source_bus = source.bus_no#ppc_int['bus'][source.bus_no,0]
            else:
                # Generators or VSC
                source_bus = ppc_int.gen.bus[source.gen_no]
                
            I[source_bus] = source.calc_currents(v_prev[source_bus])
            
        
        # Solve for network voltages
        vtmp = Ybus_inv.solve(I) 
        verr = np.abs(np.dot((vtmp-v_prev),np.transpose(vtmp-v_prev)))
        v_prev = vtmp
        i = i + 1
    
    if i >= max_iter:
        print('Network voltages and current injections did not converge in time step...')
    
    return v_prev
