#!/usr/bin/env python

import os
import math
from pyscf import gto
from pyscf import pbc
from pyscf.lib import logger

class Cell(pbc.gto.cell.Cell):
    '''A class for NEO with periodic boundary condition.
    By default, all atoms would be treated quantum mechanically.

    Example:

    >>> from pyscf import neo
    >>> cl = neo.pbc.Cell()
    >>> cl.build(a='5.21 0 0; 0 5.21 0; 0 0 5.21', atom='H 0 0 0; H 2.105 2.105 2.105',
                    basis='sto3g', quantum_nuc = [0,1])
    '''
    def __init__(self, **kwargs):
        pbc.gto.cell.Cell.__init__(self, **kwargs)
        self.quantum_nuc = [] # a list to assign which nuclei are treated quantum mechanically
        self.nuc_num = 0 # the number of quantum nuclei
        self.mass = [] # the mass of nuclei
        self.elec = None # a Mole object for NEO-electron and classical nuclei
        self.nuc = [] # a list of Mole objects for quantum nuclei
        self.mesh = [10, 10, 10] #NOTE

    def nuc_cell(self, atom_index):
        '''
        Return a Cell object for specified quantum nuclei.

        H: PB4-D  J. Chem. Phys. 152, 244123 (2020)
        D: scaled PB4-D
        other atoms: 12s12p12d, alpha=2*sqrt(2)*mass, beta=sqrt(3)
        '''

        nuc = pbc.gto.Cell() # a Cell object for quantum nuclei
        nuc.atom_index = atom_index

        dirnow = os.path.realpath(os.path.join(__file__, '..'))
        if self.atom_symbol(atom_index) == 'H@2':
            basis = gto.basis.parse(open(os.path.join(dirnow, 'basis/s-pb4d.dat')).read())
        elif self.atom_pure_symbol(atom_index) == 'H':
            basis = gto.basis.parse(open(os.path.join(dirnow, 'pb4d.dat')).read())
        else:
            # even-tempered basis
            alpha = 2 * math.sqrt(2) * self.mass[atom_index]
            beta = math.sqrt(3)
            n = 12
            basis = gto.expand_etbs([(0, n, alpha, beta), (1, n, alpha, beta), (2, n, alpha, beta)])
            #logger.info(self, 'Nuclear basis for %s: n %s alpha %s beta %s' %(self.atom_symbol(atom_index), n, alpha, beta))
        nuc.build(a = self.a, atom = self.atom, basis={self.atom_symbol(atom_index): basis},
                    charge = 0, spin=0, cart = self.cart)

        quantum_nuclear_charge = 0
        for i in range(self.natm):
            if self.quantum_nuc[i] is True:
                quantum_nuclear_charge -= nuc._atm[i,0]
                nuc._atm[i,0] = 0 # set the nuclear charge of quantum nuclei to be 0
        nuc.charge += quantum_nuclear_charge
        nuc.nelectron = 2 # avoid UHF

        return nuc

    def build(self, quantum_nuc = 'all', nuc_basis = 'etbs', **kwargs):
        'assign which nuclei are treated quantum mechanically by quantum_nuc (list)'
        pbc.gto.cell.Cell.build(self, **kwargs)

        self.quantum_nuc = [False]*self.natm

        if quantum_nuc is 'all':
            self.quantum_nuc = [True]*self.natm
            logger.info(self, 'All atoms are treated quantum-mechanically by default.')
        elif isinstance(quantum_nuc, list):
            for i in quantum_nuc:
                self.quantum_nuc[i] = True
                logger.info(self, 'The %s(%i) atom is treated quantum-mechanically' %(self.atom_symbol(i), i))
        else:
            raise TypeError('Unsupported parameter %s' %(quantum_nuc))

        self.nuc_num = len([i for i in self.quantum_nuc if i == True])

        self.mass = self.atom_mass_list(isotope_avg=True)
        for i in range(len(self.mass)):
            if self.atom_symbol(i) == 'H@2': # Deuterium (from Wikipedia)
                self.mass[i] = 2.01410177811
            elif self.atom_symbol(i) == 'H@0': # Muonium (TODO: precise mass)
                self.mass[i] = 0.114
            elif self.atom_pure_symbol(i) == 'H': # Proton (from Wikipedia)
                self.mass[i] = 1.007276466621

        # build the Cell object for electrons and classical nuclei
        self.elec = pbc.gto.Cell()
        self.elec.build(**kwargs)
        quantum_nuclear_charge = 0
        for i in range(self.natm):
            if self.quantum_nuc[i] is True:
                quantum_nuclear_charge -= self.elec._atm[i,0]
                self.elec._atm[i,0] = 0 # set the nuclear charge of quantum nuclei to be 0
        self.elec.charge += quantum_nuclear_charge # charge determines the number of electrons

        # build a list of Mole objects for quantum nuclei
        for i in range(len(self.quantum_nuc)):
            if self.quantum_nuc[i] == True:
                self.nuc.append(self.nuc_cell(i))
