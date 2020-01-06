#!/usr/bin/env python

import math
import numpy 
from pyscf import gto
from pyscf.lib import logger

class Mole(gto.mole.Mole):
    'a subclass of gto.mole.Mole to handle quantum nuclei in NEO'

    def set_quantum_nuclei(self, alist):
        'assign which nuclei are treated quantum mechanically by a list(alist)'
        self.quantum_nuc = [False]*self.natm
        if alist is not None:
            self.nuc_num = len(alist)
            for i in alist:
                if self.atom_pure_symbol(i) == 'H':
                    self.quantum_nuc[i] = True
                else:
                    raise NotImplementedError('Only support quantum H now')

        logger.note(self, 'The H atom is set to be treated quantum-mechanically')
        self.nuclei_expect_position = self.atom_coord(0) #beta
        self.mole_elec()
        self.mole_nuc()

    def set_nuclei_expect_position(self, position=None, unit='B'):
        'set an expectation value of the position operator for quantum nuclei(only support single proton now)'        
        if position == None:
            position = self.mol.atom_coord(0) #beta
        if unit == 'A':
            self.nuclei_expect_position = position*1.88973
        elif unit == 'B':
            self.nuclei_expect_position = position

    def mole_elec(self):
        'return a Mole object for NEO-electron'

        self.elec = gto.mole.copy(self)
        for i in range(self.elec.natm):
            if self.elec.quantum_nuc[i] == True:
                self.elec._atm[i,0] = 0 # set the nuclear charge of quantum nuclei to be 0

        self.elec.charge -= int(self.nuc_num) 
        return self.elec

    def mole_nuc(self):
        'return a Mole object for quantum nuclei'
        self.nuc = gto.mole.copy(self)

        alpha = 2*math.sqrt(2)
        beta = math.sqrt(2)
        self.nuc_basis = gto.expand_etbs([(0, 8, alpha, beta), (1, 8, alpha, beta), (2, 8, alpha, beta)]) # even-tempered basis 8s8p8d
        self.nuc._basis = gto.mole.format_basis({'H': self.nuc_basis}) #only support quantum H now
        self.nuc._atm, self.nuc._bas, self.nuc._env = gto.mole.make_env(self.nuc._atom,self.nuc._basis, self._env[:gto.PTR_ENV_START])
        for i in range(self.natm):
            if self.quantum_nuc[i] == True:
                self.nuc._atm[i,0] = 0 # set the nuclear charge of quantum nuclei to be 0

        #self.nuc.charge += int(self.nuc_num)
        #self.nuc.nelectron = self.nuc_num
        #self.nuc.spin = self.nuc_num
        return self.nuc


