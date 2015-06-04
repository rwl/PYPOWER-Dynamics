from pypower.case9 import case9
from pypower.ppoption import ppoption
from pypf.runpf import runpf
from pypf.casedata import convert_ppc

def main():
    ppc = case9()
    pfc = convert_ppc(ppc)
    ppopt = ppoption()
    runpf(pfc, ppopt)

if __name__ == '__main__':
    main()
    