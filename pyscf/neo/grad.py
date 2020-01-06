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
        atmlst = self.mol.quantum_nuc
        self.atmlst = [i for i in range(len(atmlst)) if atmlst[i] == False] # a list for classical nuclei

    #as_scanner = grad.rhf.as_scanner

    def grad_elec(self, atmlst=None):
        g = self.scf.mf_elec.nuc_grad_method()
        return g.grad(atmlst = atmlst)

    def hcore_deriv(self, atm_id): #beta
        mol = self.mol.nuc
        aoslices = mol.aoslice_by_atom()
        shl0, shl1, p0, p1 = aoslices[atm_id]
        with mol.with_rinv_as_nucleus(atm_id):
            vrinv = mol.intor('int1e_iprinv', comp=3) # <\nabla|1/r|>
            vrinv *= mol.atom_charge(atm_id)

        return vrinv + vrinv.transpose(0,2,1)

    def grad_jcross(self):
        'get the gradient for the cross term of Coulomb interactions between electrons and quantum nuclei'
        jcross = scf.jk.get_jk((self.mol.elec, self.mol.elec, self.mol.nuc, self.mol.nuc), self.scf.dm_nuc, scripts='ijkl,lk->ij', intor='int2e_ip1_sph', comp=3)
        return -jcross

    def grad_quantum_nuc(self):
        'JCP, ...'
        a = 2*numpy.einsum('ijk,jk->i', self.mol.nuc.intor('int1e_irp'), self.scf.dm_nuc)
        return numpy.dot(self.scf.f, a.reshape(3,3))

    def kernel(self, atmlst=None):
        'Unit: Hartree/Bohr'
        if atmlst == None:
            atmlst = range(self.mol.natm)
        de = numpy.zeros((len(atmlst), 3))

        aoslices = self.mol.aoslice_by_atom()
        jcross = self.grad_jcross()

        for k, ia in enumerate(atmlst):
            if self.mol.quantum_nuc[ia] == True:
                de[k] = self.grad_quantum_nuc()
            else:
                p0, p1 = aoslices[ia,2:]
                h1ao = self.hcore_deriv(ia)
                de[k] += numpy.einsum('xij,ij->x', h1ao, self.scf.dm_nuc)
                de[k] -= numpy.einsum('xij,ij->x', jcross[:,p0:p1], self.scf.dm_elec[p0:p1]) * 2

        grad_elec = self.grad_elec(atmlst = self.atmlst)
        de[self.atmlst] += grad_elec
        self._finalize()
        return de
    
    def _finalize(self):
        if self.verbose >= logger.NOTE:
            logger.note(self, '--------------- %s gradients ---------------',
                        self.base.__class__.__name__)
            _write(self, self.mol, self.de, self.atmlst)
            logger.note(self, '----------------------------------------------')

    def as_scanner(self):
        if isinstance(self, lib.GradScanner):
            return self

        #logger.info(self, 'Create scanner for %s', self.__class__)

        class SCF_GradScanner(self.__class__, lib.GradScanner):
            def __init__(self, g):
                lib.GradScanner.__init__(self, g)
            def __call__(self, mol_or_geom, **kwargs):
                if isinstance(mol_or_geom, gto.Mole):
                    mol = mol_or_geom
                else:
                    mol = self.mol.set_geom_(mol_or_geom, inplace=True)

                mol.set_quantum_nuclei([0])
                #geom = mol.atom_coords()
                #mol.elec.set_geom_(geom)
                #mol.nuc.set_geom_(geom)
                self.mol = self.base.mol = mol
                #print 'grad.py', self.mol.elec.atom_coords()
                mf_scanner = self.base
                e_tot = mf_scanner(mol)
                de = self.kernel(**kwargs)
                print 'e_tot', e_tot
                print 'de', de
                return e_tot, de

        return SCF_GradScanner(self)

# Inject to CDFT class
CDFT.Gradients = lib.class_as_method(Gradients)
