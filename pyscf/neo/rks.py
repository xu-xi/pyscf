#!/usr/bin/env python

'''
Non-relativistic restricted Kohn-Sham for NEO-DFT
'''
import numpy
from pyscf import dft
from pyscf import scf
from pyscf.neo.hf import HF

class KS(HF):
    '''
    Example:
    >>> from pyscf import neo
    >>> mol = neo.Mole()
    >>> mol.build(atom = 'H 0 0 0; F 0 0 0.917', basis = 'ccpvdz')
    >>> mol.set_quantum_nuclei([0])
    >>> mf = neo.KS(mol)
    >>> mf.scf()
    '''

    def __init__(self, mol):
        HF.__init__(self, mol)

        self.mf_elec = dft.RKS(mol.elec)
        self.mf_elec.init_guess = 'atom'
        self.mf_elec.get_hcore = self.get_hcore_elec
        self.mf_elec.xc = 'b3lyp'

