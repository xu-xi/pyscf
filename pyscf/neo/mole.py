#!/usr/bin/env python

import sys
import math
import numpy 
from pyscf import gto


class Mole(gto.mole.Mole):
    'a subclass of gto.mole.Mole to handle quantum nuclei in NEO'

    def set_quantum_nuclei(self,alist):
        'assign which nuclei are treated quantum mechanically by a list(alist)'
        self.quantum_nuc = [False]*self.natm
        if alist is not None:
            self.nuc_num = len(alist)
            for i in alist:
                if self.atom_pure_symbol(i) == 'H':
                    self.quantum_nuc[i] = True
                else:
                    print 'ERROR: only support quantum H now';sys.exit(1)

        #logging.info
        self.mole_elec()
        self.mole_nuc()

    def mole_elec(self):
        'return an Mole object for NEO-electron'

        self.elec = gto.mole.copy(self)
        for i in range(self.elec.natm):
            if self.elec.quantum_nuc[i] == True:
                self.elec._atm[i,0] = 0 # set the nuclear charge of quantum nuclei to be 0

        self.elec.charge -= int(self.nuc_num) 
        return self.elec


    def mole_nuc(self):
        'return an Mole object for quantum nuclei'
        self.nuc = gto.mole.copy(self)

        alpha = 2*math.sqrt(2)
        beta = math.sqrt(2)
        self.nuc_basis = gto.expand_etbs([(0, 8, alpha, beta), (1, 8, alpha, beta), (2, 8, alpha, beta)]) # even-tempered basis 8s8p8d
        self.nuc._basis = gto.mole.format_basis({'H': self.nuc_basis}) #only support hydrogen now
        self.nuc._atm, self.nuc._bas, self.nuc._env = gto.mole.make_env(self.nuc._atom,self.nuc._basis)
        for i in range(self.natm):
            if self.quantum_nuc[i] == True:
                self.nuc._atm[i,0] = 0 # set the nuclear charge of quantum nuclei to be 0

        #self.nuc.charge += int(self.nuc_num)
        return self.nuc


