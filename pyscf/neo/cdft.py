#!/usr/bin/env python

'''
Constrained nuclear-electronic orbital density functional theory
'''
import numpy
from pyscf import scf
from pyscf.neo.ks import KS


class CDFT(KS):
    '''
    Example:

    >>> mol = neo.Mole()
    >>> mol.build(atom = 'H 0.0 0.0 0.0; C 0.0 0.0 1.064; N 0.0 0.0 2.220', basis = 'ccpvdz')
    >>> mf = neo.CDFT(mol)
    >>> mf.scf()
    '''

    def __init__(self, mol, **kwargs):
        self.f = [numpy.zeros(3)] * mol.natm
        KS.__init__(self, mol, **kwargs)
        #self.scf = self.inner_scf

        # set up the Hamiltonian for each quantum nuclei in cNEO
        for i in range(len(self.mol.nuc)):
            mf = self.mf_nuc[i]
            mf.nuclei_expect_position = mf.mol.atom_coord(mf.mol.atom_index)
            mf.get_hcore = self.get_hcore_nuc

    def get_hcore_nuc(self, mol):
        'get the core Hamiltonian for quantum nucleus in cNEO'
        h = super().get_hcore_nuc(mol)

        # extra term in cNEO due to the constraint on expectational position
        ia = mol.atom_index
        h += numpy.einsum('xij,x->ij', mol.intor_symmetric('int1e_r', comp=3), self.f[ia])

        return h

    def first_order_de(self, f, mf):
        'The first order derivative of L w.r.t the Lagrange multiplier f'
        i = self.mf_nuc.index(mf)
        ia = mf.mol.atom_index
        self.f[ia] = f
        h1 = mf.get_hcore(mol = mf.mol)
        veff = mf.get_veff(mf.mol, self.dm_nuc[i])
        s1n = mf.get_ovlp()
        energy, coeff = mf.eig(h1 + veff, s1n)
        occ = mf.get_occ(energy, coeff)
        self.dm_nuc[i] = scf.hf.make_rdm1(coeff, occ)

        return numpy.einsum('xij,ji->x', mf.mol.intor_symmetric('int1e_r', comp=3), self.dm_nuc[i]) - mf.nuclei_expect_position

    def L_second_order(self, mf_nuc):
        '(not used) The (approximate) second-order derivative of L w.r.t f'

        energy = mf_nuc.mo_energy
        coeff = mf_nuc.mo_coeff

        ints = mf_nuc.mol.intor_symmetric('int1e_r', comp=3)

        de = 1.0/(energy[0] - energy[1:])

        ints = numpy.einsum('...pq,p,qj->...j', ints, coeff[:,0].conj(), coeff[:,1:])
        return 2*numpy.einsum('ij,lj,j->il', ints, ints.conj(), de).real

    def energy_qmnuc(self, mf_nuc, h1n, dm_nuc):
        ia = mf_nuc.mol.atom_index
        h_r = numpy.einsum('xij,x->ij', mf_nuc.mol.intor_symmetric('int1e_r', comp=3), self.f[ia])
        e = super().energy_qmnuc(mf_nuc, h1n, dm_nuc) - numpy.einsum('ij,ji', h_r, dm_nuc)
        return e

    def nuc_grad_method(self):
        from pyscf.neo.grad import Gradients
        return Gradients(self)
