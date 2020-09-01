#!/usr/bin/env python

'''
Analytical Hessian for constrained nuclear-electronic orbitals
'''
import numpy
from pyscf import lib
from pyscf import hessian
from pyscf import scf
from pyscf.scf import cphf
from functools import reduce

class Hessian(lib.StreamObject):
    '''
    Example:

    >>> mol = neo.Mole()
    >>> mol.build(atom = 'H 0 0 0.00; C 0 0 1.064; N 0 0 2.220', basis = 'ccpvdz')
    >>> mf = neo.CDFT(mol)
    >>> mf.mf_elec.xc = 'b3lyp'
    >>> mf.scf()

    >>> g = neo.Hessian(mf)
    >>> g.kernel()
    '''

    def __init__(self, scf_method):
        self.mol = scf_method.mol
        self.base = scf_method
        self.atmlst = range(self.mol.natm)
        self.de = numpy.zeros((0, 0, 3, 3))

    def hess_elec(self, mo_e1=None):
        'Hessian of electrons and classic nuclei in NEO'
        hobj = self.base.mf_elec.Hessian() 
        return hobj.hess_elec(mo_e1) + hobj.hess_nuc()
 
    def get_hcore(self, mol):
        'Part of the second derivatives of core Hamiltonian of quantum nuclei'
        i = mol.atom_index
        mass = 1836.15267343 * self.mol.mass[i]
        h1aa = mol.intor('int1e_ipipkin', comp=9)/mass
        h1ab = mol.intor('int1e_ipkinip', comp=9)/mass

        if mol._pseudo:
            NotImplementedError('Nuclear hessian for GTH PP')
        else:
            h1aa += mol.intor('int1e_ipipnuc', comp=9) * self.mol._atm[i,0]
            h1ab += mol.intor('int1e_ipnucip', comp=9) * self.mol._atm[i,0]
        if mol.has_ecp():
            NotImplementedError('Nuclear hessian for ECPscalar')
        #    h1aa += mol.intor('ECPscalar_ipipnuc', comp=9)
        #    h1ab += mol.intor('ECPscalar_ipnucip', comp=9)
        nao = h1aa.shape[-1]
        return h1aa.reshape(3,3,nao,nao), h1ab.reshape(3,3,nao,nao)

    def get_ovlp(self, mol):
        s1a =-mol.intor('int1e_ipovlp', comp=3)
        nao = s1a.shape[-1]
        s1aa = mol.intor('int1e_ipipovlp', comp=9).reshape(3,3,nao,nao)
        s1ab = mol.intor('int1e_ipovlpip', comp=9).reshape(3,3,nao,nao)
        return s1aa, s1ab, s1a

    def ao2mo(self, mf, mat):
        return numpy.asarray([reduce(numpy.dot, (mf.mo_coeff.T, x, mf.mo_coeff[:,mf.mo_occ>0])) for x in mat])


    def kernel(self):
        helec = self.hess_elec()
        print(helec)


from pyscf.neo import CDFT
CDFT.Hessian = lib.class_as_method(Hessian)


