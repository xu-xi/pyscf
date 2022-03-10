#!/usr/bin/env python

'''
Nuclear Electronic Orbital Hartree-Fock (NEO-HF) with periodic boundary condition
'''

import numpy
import scipy
from pyscf import scf
from pyscf import neo
from pyscf import pbc
from pyscf.lib import logger
from pyscf.data import nist

class KHF(pbc.scf.khf.KRHF):
    '''
    Hartree Fock for NEO with periodic boundary condition

    Example:

    >>> from pyscf import neo
    >>> cl = neo.pbc.Cell()
    >>> cl.build(a = '5.21 0 0; 0 5.21 0; 0 0 5.21', atom = 'H 0 0 0; H 2.105 2.105 2.105',
                    basis = 'sto3g', quantum_nuc=[0,1])
    >>> mf = neo.pbc.KHF(cl, kpts=[2,2,2])
    >>> mf.scf()
    '''

    def __init__(self, cell):
        pbc.scf.khf.KRHF.__init__(self, cell)

    def get_hcore_nuc(self, cell, kpts):
        'get the core Hamiltonian for quantum nucleus.'

        ia = cell.atom_index
        mass = self.mol.mass[ia] * nist.ATOMIC_MASS/nist.E_MASS # the mass of quantum nucleus in a.u.
        charge = self.mol.atom_charge(ia)

        # nuclear kinetic energy and Coulomb interactions with classical nuclei
        h = cell.pbc_intor('int1e_kin', hermi=1, kpts=np.zeros(3))/mass
        h -= cell.pbc_intor('int1e_nuc', hermi=1, kpts=np.zeros(3))*charge

        # Coulomb interactions between quantum nucleus and electrons
        h -= pbc.gto.cell.get_jk((mole, mole, self.mol.elec, self.mol.elec),
                        self.dm_elec, scripts='ijkl,lk->ij', aosym ='s4') * charge

        # Coulomb interactions between quantum nuclei
        for j in range(len(self.dm_nuc)):
            ja = self.mol.nuc[j].atom_index
            if ja != ia and isinstance(self.dm_nuc[j], numpy.ndarray):
                h += scf.jk.get_jk((mole, mole, self.mol.nuc[j], self.mol.nuc[j]),
                                   self.dm_nuc[j], scripts='ijkl,lk->ij')*charge*self.mol.atom_charge(ja)

        return h
