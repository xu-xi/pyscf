#!/usr/bin/env python

'''
Non-relativistic restricted Kohn-Sham for NEO-DFT
'''
import numpy
from pyscf import dft
from pyscf import scf
from pyscf.neo.hf import HF

class KS(HF):
    '''
    Example:
    >>> from pyscf import neo
    >>> mol = neo.Mole()
    >>> mol.build(atom = 'H 0 0 0; F 0 0 0.917', basis = 'ccpvdz')
    >>> mol.set_quantum_nuclei([0])
    >>> mf = neo.KS(mol)
    >>> mf.scf()
    '''

    def __init__(self, mol):
        HF.__init__(self, mol)
        self.xc = 'HF'

    def energy_tot(self, ks, dm_elec, dm_nuc):
        'total energy'
        mol = self.mol

        energy_classical_nuc = mol.elec.energy_nuc()

        h1e = scf.hf.get_hcore(mol.elec)
        E1_elec = numpy.einsum('ij,ji', h1e, dm_elec)
        vhf_elec = dft.rks.get_veff(ks, mol.elec, dm_elec)
        jcross = self.elec_nuc_coulomb(dm_elec, dm_nuc)

        E_coul_elec = vhf_elec.ecoul + vhf_elec.exc - numpy.einsum('ij,ji', jcross, dm_elec)*0.5

        h1n = self.get_hcore_nuc()
        E1_nuc = numpy.einsum('ij,ji', h1n, dm_nuc)
        vhf_nuc = self.get_veff_nuc(dm_elec, dm_nuc)
        E_coul_nuc = numpy.einsum('ij,ji', vhf_nuc, dm_nuc)*0.5

        print energy_classical_nuc, E1_elec, E_coul_elec, E1_nuc, E_coul_nuc

        return energy_classical_nuc + E1_elec + E_coul_elec + E1_nuc + E_coul_nuc #double count now
    def scf_test(self, conv_tot = 1e-7):
        mol = self.mol
        max_cycle = 100
        self.dm_nuc = self.init_guess_by_core_hamiltonian()

        mf_elec = dft.RKS(mol.elec)
        mf_elec.xc = self.xc
        mf_elec.init_guess = 'atom'
        mf_elec.get_hcore = self.get_hcore_elec
        mf_elec.kernel()
        self.dm_elec = scf.hf.make_rdm1(mf_elec.mo_coeff, mf_elec.mo_occ)

        h1n = self.get_hcore_nuc()
        s1n = scf.hf.get_ovlp(mol.nuc)
        vhf_nuc = self.get_veff_nuc(self.dm_elec, self.dm_nuc)
        fock_nuc = h1n + vhf_nuc
        no_energy, no_coeff = scf.hf.eig(fock_nuc, s1n)
        no_occ = numpy.zeros(len(no_energy))
        no_occ[:mol.nuc_num] = 1 #high-spin quantum nuclei
        self.dm_nuc = scf.hf.make_rdm1(no_coeff, no_occ)

        E_tot = self.energy_tot2(mf_elec, self.dm_nuc)
        print 'Initial energy:', E_tot
        scf_conv = False
        cycle = 0

        while not scf_conv and cycle <= max_cycle:
            cycle += 1
            E_last = E_tot

            mf_elec.kernel()
            self.dm_elec = scf.hf.make_rdm1(mf_elec.mo_coeff, mf_elec.mo_occ)

            vhf_nuc = self.get_veff_nuc(self.dm_elec, self.dm_nuc)

            fock_nuc = h1n + vhf_nuc
            no_energy, no_coeff = scf.hf.eig(fock_nuc, s1n)
            no_occ = numpy.zeros(len(no_energy))
            no_occ[:mol.nuc_num] = 1 #high-spin quantum nuclei
            self.dm_nuc = scf.hf.make_rdm1(no_coeff, no_occ)

            E_tot = self.energy_tot2(mf_elec, self.dm_nuc)
            print 'Cycle',cycle
            print E_tot

            if abs(E_tot - E_last) < conv_tot:
                scf_conv = True
                print 'Converged'

    def scf(self, conv_tot = 1e-7):
        mol = self.mol

        dm_elec = scf.hf.init_guess_by_atom(mol.elec)
        dm_nuc = self.init_guess_by_core_hamiltonian()

        mf_elec = dft.RKS(mol.elec)
        mf_elec.init_guess = 'atom'
        mf_elec.xc = self.xc

        #mf_elec.kernel()

        h1e = scf.hf.get_hcore(mol.elec)
        s1e = scf.hf.get_ovlp(mol.elec)

        h1n = self.get_hcore_nuc()
        s1n = scf.hf.get_ovlp(mol.nuc)

        vhf_elec = dft.rks.get_veff(mf_elec, mol.elec, dm_elec) - self.elec_nuc_coulomb(dm_elec, dm_nuc)
        vhf_nuc = self.get_veff_nuc(dm_elec, dm_nuc)

        E_tot = self.energy_tot(mf_elec, dm_elec, dm_nuc)
        print 'Initial energy:',E_tot
        
        cycle = 0
        scf_conv = False
        max_cycle = 100

        while not scf_conv and cycle < max_cycle:
            cycle += 1
            E_last = E_tot

            #fock_elec = self.get_fock(h1e, s1e, vhf_elec, dm_elec, cycle)
            fock_elec = h1e + vhf_elec
            eo_energy, eo_coeff = scf.hf.eig(fock_elec, s1e)
            eo_occ = self.get_occ(eo_energy, eo_coeff)
            dm_elec = self.make_rdm1(eo_coeff, eo_occ)

            vhf_nuc = self.get_veff_nuc(dm_elec, dm_nuc)

            #fock_nuc = self.get_fock(h1n, s1n, vhf_nuc, dm_nuc, cycle)
            fock_nuc = h1n + vhf_nuc
            no_energy, no_coeff = scf.hf.eig(fock_nuc, s1n)
            no_occ = numpy.zeros(len(no_energy))
            no_occ[:mol.nuc_num] = 1 #high-spin quantum nuclei
            dm_nuc = scf.hf.make_rdm1(no_coeff, no_occ)

            vhf_elec = dft.rks.get_veff(mf_elec, mol.elec, dm_elec) - self.elec_nuc_coulomb(dm_elec, dm_nuc)

            E_tot = self.energy_tot(mf_elec, dm_elec, dm_nuc)
            print 'Cycle',cycle
            print E_tot

            if abs(E_tot - E_last) < conv_tot:
                scf_conv = True
                print 'Converged'





