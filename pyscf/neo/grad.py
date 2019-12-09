#!/usr/bin/env python

'''
Analytical nuclear gradient for constrained nuclear-electronic orbital
'''

class Gradients():
    def __init__(self, scf_method):
        self.mol = scf_method.mol
        self.base = scf_method
        atmlst = self.mol.quantum_nuc
        self.atmlst = [i for i in range(len(atmlst)) if atmlst[i] == False]

    def grad_elec(self):
        g = self.base.mf_elec.nuc_grad_method()
        g.grad(atmlst = self.atmlst)

    def grad_hcore_nuc(self):
        mass_proton = 1836.15267343
        h = self.mol.nuc.intor('int1e_ipkin', comp=3)/mass_proton
        h += self.mol.nuc.intor('int1e_ipnuc', comp=3)
        return -h

    def grad_ovlp(self):
        return -self.mol.nuc.intor('int1e_ipovlp', comp=3)
    
    def make_rdm1e(self):
        mo_energy = self.base.mf_nuc.mo_energy
        mo_coeff = self.base.mf_nuc.mo_coeff
        mo_occ = self.base.mf_nuc.occ

        mo0 = mo_coeff[:,mo_occ>0]
        mo0e = mo0 * (mo_energy[mo_occ>0] * mo_occ[mo_occ>0])
        return numpy.dot(mo0e, mo0.T.conj())

    def grad_veff_nuc(self):
        pass

    def grad_jcross(self):
        'get the gradient for the cross term of Coulomb interactions between electrons and quantum nuclei'
        pass



    def grad_quantum_nuc(self):
        pass


