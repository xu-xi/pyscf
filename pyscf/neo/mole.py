#!/usr/bin/env python

import math
import numpy 
from pyscf import gto
from pyscf.lib import logger

class Mole(gto.mole.Mole):
    '''A subclass of gto.mole.Mole to handle quantum nuclei in NEO.
    By default, all atoms would be treated quantum mechanically.

    Example:

    >>> from pyscf import neo
    >>> mol = neo.Mole()
    >>> mol.build(atom = 'H 0 0 0; C 0 0 1.1; N 0 0 2.2', basis = 'ccpvdz')
    >>> mol.set_quantum_nuclei() # all atoms are treated quantum mechanically
    >>> mol.set_quantum_nuclei([0, 1]) # H and C are treated quantum mechanically

    '''

    def __init__(self, **kwargs):
        gto.mole.Mole.__init__(self, **kwargs)

    def elec_mole(self):
        'return a Mole object for NEO-electron and classical nuclei'

        eole = gto.mole.copy(self) # a Mole object for electrons
        quantum_nuclear_charge = 0
        for i in range(self.natm):
            if self.quantum_nuc[i] == True:
                quantum_nuclear_charge -= eole._atm[i,0]
                eole._atm[i,0] = 0 # set the nuclear charge of quantum nuclei to be 0
        eole.charge += quantum_nuclear_charge
        return eole

    def nuc_mole(self, atom_index, basis='etbs'):
        'return a Mole object for specified quantum nuclei, the default basis is even-tempered Gaussian basis'
        nole = gto.mole.copy(self) # a Mole object for quantum nuclei
        nole.atom_index = atom_index

        alpha = 2*math.sqrt(2)
        beta = math.sqrt(2)

        if basis == 'etbs':
            basis = gto.expand_etbs([(0, 8, alpha, beta), (1, 8, alpha, beta), (2, 8, alpha, beta)]) # even-tempered basis 8s8p8d
        nole._basis = gto.mole.format_basis({self.atom_symbol(atom_index): basis})
        nole._atm, nole._bas, nole._env = gto.mole.make_env(nole._atom, nole._basis, self._env[:gto.PTR_ENV_START])
        for i in range(self.elec.natm):
            if self.quantum_nuc[i] == True:
                nole._atm[i,0] = 0 # set the nuclear charge of quantum nuclei to be 0

        #self.nuc.charge += int(self.nuc_num)
        #self.nuc.nelectron = self.nuc_num
        #self.nuc.spin = self.nuc_num
        return nole

    def build(self, alist = 'all', **kwargs):
        'assign which nuclei are treated quantum mechanically by a list(alist)'
        gto.mole.Mole.build(self, **kwargs)

        self.quantum_nuc = [False]*self.natm

        if alist == 'all':
            self.quantum_nuc = [True]*self.natm
            logger.note(self, 'All atoms are treated quantum-mechanically by default.')
        elif isinstance(alist, list):
            for i in alist:
                self.quantum_nuc[i] = True
                logger.note(self, 'The %s(%i) atom is treated quantum-mechanically' %(self.atom_symbol(i), i))
        else:
            raise NotImplementedError('Unsupported parameter %s' %(alist))

        self.nuc_num = len([i for i in self.quantum_nuc if i == True]) # the number of quantum nuclei
        self.elec = self.elec_mole()
        self.nuc = []
        for i in range(len(self.quantum_nuc)):
            if self.quantum_nuc[i] == True:
                self.nuc.append(self.nuc_mole(i))

    def set_nuclei_expect_position(self, position=None, unit='B'):
        'set an expectation value of the position operator for quantum nuclei(only support single proton now)'        
        if position == None:
            position = self.mol.atom_coord(0) #beta
        if unit == 'A':
            self.nuclei_expect_position = position*1.88973
        elif unit == 'B':
            self.nuclei_expect_position = position


