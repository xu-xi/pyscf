#!/usr/bin/env python

'''
Nuclear Electronic Orbital Hartree Fock (NEO-HF)
'''

import numpy
from pyscf import gto
from pyscf import scf
from pyscf import lib
from pyscf.scf.hf import SCF


class HF(SCF):
    'Hartree Fock for NEO'
    def __init__(self, mol):
        SCF.__init__(self, mol)

    def get_hcore_nuc(self):
        'get core Hamiltonian for quantum nuclei'
        #Z = mol._atm[:,0] # nuclear charge
        #M = gto.mole.atom_mass_list(mol)*1836 # Note: proton mass
        mol = self.mol
        mol.mole_nuc()
        h = gto.moleintor.getints('int1e_kin_sph', mol.nuc._atm, mol.nuc._bas, mol.nuc._env, hermi=1, aosym='s4')/1836.1527
        h -= gto.moleintor.getints('int1e_nuc_sph', mol.nuc._atm, mol.nuc._bas, mol.nuc._env, hermi=1, aosym='s4')

        return h

    def init_guess_by_core_hamiltonian(self):
        '''Generate initial guess density matrix for quantum nuclei from core hamiltonian

           Returns:
            Density matrix, 2D ndarray
        '''
        mol = self.mol
        h1n = self.get_hcore_nuc()
        s1n = scf.hf.get_ovlp(mol.nuc)
        nuc_energy, nuc_coeff = scf.hf.eig(h1n, s1n)
        nuc_occ = numpy.zeros(len(nuc_energy))
        nuc_occ[:mol.nuc_num] = 1 #high-spin quantum nuclei

        return scf.hf.make_rdm1(nuc_coeff, nuc_occ)

    def get_veff_elec(self, dm_elec, dm_nuc):
        'get the HF effective potential for electrons in NEO'
        mol = self.mol
        #vj, vk = scf.hf.get_jk(mol.elec, dm_elec)
        vj, vk = scf.jk.get_jk(mol.elec, (dm_elec,dm_elec), ('ijkl,ji->kl','ijkl,jk->il'), aosym='s8')
        jcross = scf.jk.get_jk((mol.elec, mol.elec, mol.nuc, mol.nuc), dm_nuc, scripts='ijkl,lk->ij', aosym ='s4')
        
        return vj - vk * .5 - jcross 

    def get_veff_nuc(self, dm_elec, dm_nuc):
        'get the HF effective potential for quantum nuclei in NEO'
        mol = self.mol
        #vj , vk = scf.hf.get_jk(mol.nuc, dm_nuc)
        vj, vk = scf.jk.get_jk(mol.nuc, (dm_nuc,dm_nuc), ('ijkl,ji->kl','ijkl,jk->il'), aosym='s8')
        #inte2 = gto.moleintor.getints('int2e_sph',mol.nuc._atm, mol.nuc._bas, mol.nuc._env, aosym='s8').reshape((mol.nuc.nao_nr(),)*4)
        #vj = numpy.einsum('ijkl,ji->kl', inte2, dm_nuc)
        #vk = numpy.einsum('ijkl,jk->il', inte2, dm_nuc)

        jcross = scf.jk.get_jk((mol.nuc, mol.nuc, mol.elec, mol.elec), dm_elec, scripts='ijkl,lk->ij', aosym = 's4')
        #print numpy.linalg.norm(vj),numpy.linalg.norm(vk)
        return vj - vk - jcross
        #return -jcross

    def elec_nuc_coulomb(self, dm_elec, dm_nuc):
        'get the Coulomb energy between electrons and quantum nuclei'
        mol = self.mol
        #jcross = scf.jk.get_jk((mol.nuc, mol.nuc, mol.elec, mol.elec), dm_elec, scripts='ijkl,lk->ij', aosym = 's4')
        jcross = scf.jk.get_jk((mol.elec, mol.elec, mol.nuc, mol.nuc), dm_nuc, scripts='ijkl,lk->ij', aosym = 's4')
        ecoul = numpy.einsum('ij,ij', jcross, dm_elec)
        #return jcross
        return ecoul

    def HF_energy_tot(self, dm_elec, dm_nuc):
        'Total HF energy of NEO'
        mol = self.mol

        energy_classical_nuc = mol.elec.energy_nuc()

        h1e = scf.hf.get_hcore(mol.elec)
        E1_elec = numpy.einsum('ij,ji', h1e, dm_elec)
        vhf_elec = self.get_veff_elec(dm_elec, dm_nuc)
        #vhf_elec = scf.hf.get_veff(mol.elec, dm_elec)
        E_coul_elec = numpy.einsum('ij,ji', vhf_elec, dm_elec) * 0.5

        h1n = self.get_hcore_nuc()
        E1_nuc = numpy.einsum('ij,ji', self.get_hcore_nuc(), dm_nuc)
        vhf_nuc = self.get_veff_nuc(dm_elec, dm_nuc)
        #vhf_nuc = scf.hf.get_veff(mol.nuc, dm_nuc)
        E_coul_nuc = numpy.einsum('ij,ji', vhf_nuc, dm_nuc)* 0.5

        print energy_classical_nuc, E1_elec, E_coul_elec, E1_nuc, E_coul_nuc

        E_tot = energy_classical_nuc  + E1_elec + E_coul_elec  + E1_nuc + E_coul_nuc 
        
        return E_tot


    def scf(self, conv_tot = 1e-10):
        'self-consistent field'
        mol = self.mol

        dm_elec = scf.hf.init_guess_by_1e(mol.elec)
        dm_nuc = self.init_guess_by_core_hamiltonian()

        h1e = scf.hf.get_hcore(mol.elec)
        s1e = scf.hf.get_ovlp(mol.elec)

        h1n = self.get_hcore_nuc()
        s1n = scf.hf.get_ovlp(mol.nuc)

        vhf_elec = self.get_veff_elec(dm_elec, dm_nuc)
        vhf_nuc = self.get_veff_nuc(dm_elec, dm_nuc)

        E_tot =  self.HF_energy_tot(dm_elec, dm_nuc)
        print 'Initial energy:',E_tot
        cycle = 0
        scf_conv = False
        max_cycle = 100

        while not scf_conv and cycle <= max_cycle:
            E_last = E_tot
            
            #fock_elec = self.get_fock(h1e, s1e, vhf_elec, dm_elec, cycle)
            fock_elec = h1e + vhf_elec

            eo_energy, eo_coeff = scf.hf.eig(fock_elec, s1e)
            eo_occ = self.get_occ(eo_energy, eo_coeff)
            dm_elec = self.make_rdm1(eo_coeff, eo_occ)
            print 'norm_elec',numpy.linalg.norm(dm_elec)
            #dm_elec = lib.tag_array(dm_elec, mo_coeff = eo_coeff, mo_occ = eo_occ)
            vhf_nuc = self.get_veff_nuc(dm_elec, dm_nuc)

            #fock_nuc = self.get_fock(h1n, s1n, vhf_nuc, dm_nuc, cycle)
            fock_nuc = h1n + vhf_nuc
            no_energy, no_coeff = scf.hf.eig(fock_nuc, s1n)
            no_occ = numpy.zeros(len(no_energy))
            no_occ[:mol.nuc_num] = 1 #high-spin quantum nuclei 
            dm_nuc = scf.hf.make_rdm1(no_coeff, no_occ)
            #dm_nuc = lib.tag_array(dm_nuc, mo_coeff = no_coeff, mo_occ = no_occ)
            print 'norm_nuc',numpy.linalg.norm(dm_nuc)
            print eo_occ
            print eo_energy
            print no_occ
            print no_energy

            vhf_elec = self.get_veff_elec(dm_elec, dm_nuc)

            E_tot = self.HF_energy_tot(dm_elec, dm_nuc)
            print 'Cycle',cycle
            print E_tot
            cycle += 1

            if abs(E_tot - E_last) < conv_tot:
                scf_conv = True

            if scf_conv:
                print scf_conv
                print E_last
                print eo_occ
                print eo_energy
                print no_occ
                print no_energy

