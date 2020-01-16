#!/usr/bin/env python

'''
Analytical nuclear gradient for constrained nuclear-electronic orbital
'''
import numpy
from pyscf import gto
from pyscf import scf
from pyscf import grad
from pyscf import lib
from pyscf.lib import logger
from pyscf.neo import CDFT
from pyscf.grad.rhf import _write

class Gradients(lib.StreamObject):
    '''
    Example:

    >>> mol = neo.Mole()
    >>> mol.build(atom = 'H 0 0 0.00; C 0 0 1.064; N 0 0 2.220', basis = 'ccpvtz')
    >>> mol.set_quantum_nuclei([0])
    >>> mol.set_nuclei_expect_position(mol.atom_coord(0), unit='B')
    >>> mf = neo.CDFT(mol)
    >>> mf.mf_elec.xc = 'b3lyp'
    >>> mf.inner_scf()

    >>> g = neo.Gradients(mf)
    >>> g.kernel()
    '''

    def __init__(self, scf_method):
        self.mol = scf_method.mol
        self.base = scf_method
        self.scf = scf_method
        self.verbose = 4
        atmlst = self.mol.quantum_nuc
        self.atmlst = [i for i in range(len(atmlst)) if atmlst[i] == False] # a list for classical nuclei
        self.grad = self.kernel

    #as_scanner = grad.rhf.as_scanner

    def grad_elec(self, atmlst=None):
        g = self.scf.mf_elec.nuc_grad_method()
        g.verbose = 2
        return g.grad(atmlst = atmlst)

    def get_hcore(self):
        mass_proton = 1836.15267343
        h = self.mol.nuc.intor('int1e_ipkin', comp=3)/mass_proton
        h -= self.mol.nuc.intor('int1e_ipnuc', comp=3)
        return h

    def hcore_deriv(self, atm_id): #beta
        mol = self.mol.nuc
        with mol.with_rinv_as_nucleus(atm_id):
            vrinv = mol.intor('int1e_iprinv', comp=3) # <\nabla|1/r|>
            vrinv *= -mol.atom_charge(atm_id)

        return vrinv + vrinv.transpose(0,2,1)

    def grad_jcross_elec(self):
        'get the gradient for the cross term of Coulomb interactions between electrons and quantum nuclei'
        jcross = scf.jk.get_jk((self.mol.elec, self.mol.elec, self.mol.nuc, self.mol.nuc), self.scf.dm_nuc, scripts='ijkl,lk->ij', intor='int2e_ip1_sph', comp=3)
        return jcross

    def grad_jcross_nuc(self):
        'get the gradient for the cross term of Coulomb interactions between electrons and quantum nuclei'
        jcross = scf.jk.get_jk((self.mol.nuc, self.mol.nuc, self.mol.elec, self.mol.elec), self.scf.dm_elec, scripts='ijkl,lk->ij', intor='int2e_ip1_sph', comp=3)
        return jcross

    def get_ovlp(self):
        return self.mol.nuc.intor('int1e_ipovlp', comp=3)

    def make_rdm1e(self):
        mo_energy = self.scf.mf_nuc.mo_energy
        mo_coeff = self.scf.mf_nuc.mo_coeff
        mo_occ = self.scf.mf_nuc.get_occ(mo_energy, mo_coeff)
        mo0 = mo_coeff[:,mo_occ>0]
        mo0e = mo0 * (mo_energy[mo_occ>0] * mo_occ[mo_occ>0])
        return numpy.dot(mo0e, mo0.T.conj())

    def get_veff(self):
        vj, vk = scf.jk.get_jk(self.mol.nuc, (self.scf.dm_nuc, self.scf.dm_nuc), ('ijkl,ji->kl','ijkl,li->kj'), intor='int2e_ip1_sph', comp=3)
        return vj - vk

    def kernel(self, atmlst=None):
        'Unit: Hartree/Bohr'
        if atmlst == None:
            atmlst = range(self.mol.natm)

        self.de = numpy.zeros((len(atmlst), 3))
        aoslices = self.mol.aoslice_by_atom()

        for k, ia in enumerate(atmlst):
            p0, p1 = aoslices[ia,2:]
            h1ao = self.hcore_deriv(ia)
            self.de[k] += numpy.einsum('xij,ij->x', h1ao, self.scf.dm_nuc)
            self.de[k] += numpy.einsum('xij,ij->x', self.get_hcore(), self.scf.dm_nuc)*2
            self.de[k] += numpy.einsum('xij,ij->x', self.get_veff(), self.scf.dm_nuc)*2
            self.de[k] -= numpy.einsum('xij,ij->x', self.get_ovlp(), self.make_rdm1e())*2
            f_deriv = numpy.einsum('ijk,jk->i', self.mol.nuc.intor('int1e_irp'), self.scf.dm_nuc)*2
            self.de[k] += numpy.dot(f_deriv.reshape(3,3), self.scf.f)
            jcross1 = self.grad_jcross_nuc()
            jcross2 = self.grad_jcross_elec()
            self.de[k] -= numpy.einsum('xij,ij->x', jcross1, self.scf.dm_nuc)*2
            self.de[k] -= numpy.einsum('xij,ij->x', jcross2[:,p0:p1], self.scf.dm_elec[p0:p1])*2

            if self.mol.quantum_nuc[ia] == True:
                self.de[k] += self.scf.f

        grad_elec = self.grad_elec()
        self.de = grad_elec - self.de
        self._finalize()
        return self.de
    
    def _finalize(self):
        if self.verbose >= logger.NOTE:
            logger.note(self, '--------------- %s gradients ---------------',
                        self.base.__class__.__name__)
            _write(self, self.mol, self.de, None)
            logger.note(self, '----------------------------------------------')

    def as_scanner(self):
        if isinstance(self, lib.GradScanner):
            return self

        logger.info(self, 'Create scanner for %s', self.__class__)

        class SCF_GradScanner(self.__class__, lib.GradScanner):
            def __init__(self, g):
                lib.GradScanner.__init__(self, g)
            def __call__(self, mol_or_geom, **kwargs):
                if isinstance(mol_or_geom, gto.Mole):
                    mol = mol_or_geom
                else:
                    mol = self.mol.set_geom_(mol_or_geom, inplace=True)

                mol.set_quantum_nuclei([0])
                self.mol = self.base.mol = mol
                mf_scanner = self.base
                e_tot = mf_scanner(mol)
                de = self.kernel(**kwargs)
                return e_tot, de

        return SCF_GradScanner(self)

# Inject to CDFT class
CDFT.Gradients = lib.class_as_method(Gradients)
