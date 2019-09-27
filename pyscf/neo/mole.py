#!/usr/bin/env python

from pyscf.gto.mole import Mole


class Mole_neo(Mole):
    'a subclass of Mole to handle quantum nuclei in NEO'
    def __init__(self):
        self.quantum_nuc = [False]*self.natm

    def set_quantum_nuc(self,alist):
        'set which nuclei are treated quantum mechanically'
        for i in alist:
            self.quantum_nuc[i] = True

        return self.quantum_nuc
