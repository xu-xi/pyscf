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
    '''Hartree Fock for NEO
    
    Example:
    
    >>> from pyscf import neo
    >>> mol = neo.Mole()
    >>> mol.build(atom = 'H 0 0 0; F 0 0 0.917', basis = 'ccpvdz')
    >>> mol.set_quantum_nuclei([0])
    >>> mf = neo.HF(mol)
    >>> mf.scf()
    
    '''

    def __init__(self, mol):
        SCF.__init__(self, mol)
        self.direct = True #direct diagonalization for 'veff' of single proton


    def get_hcore_nuc(self):
        'get core Hamiltonian for quantum nuclei'
        #Z = mol._atm[:,0] # nuclear charge
        #M = gto.mole.atom_mass_list(mol)*1836 # Note: proton mass
        mol = self.mol
        mol.mole_nuc()
        h = gto.moleintor.getints('int1e_kin_sph', mol.nuc._atm, mol.nuc._bas, mol.nuc._env, hermi=1, aosym='s4')/1836.15267343
        h -= gto.moleintor.getints('int1e_nuc_sph', mol.nuc._atm, mol.nuc._bas, mol.nuc._env, hermi=1, aosym='s4')

        return h

    def init_guess_by_core_hamiltonian(self, mol=None):
        '''Generate initial guess density matrix for quantum nuclei from core hamiltonian

           Returns:
            Density matrix, 2D ndarray
        '''
        if mol == None:
            mol = self.mol
        h1n = self.get_hcore_nuc()
        s1n = scf.hf.get_ovlp(mol.nuc)
        nuc_energy, nuc_coeff = scf.hf.eig(h1n, s1n)
        nuc_occ = numpy.zeros(len(nuc_energy))
        nuc_occ[:mol.nuc_num] = 1 #high-spin quantum nuclei

        return scf.hf.make_rdm1(nuc_coeff, nuc_occ)

    def get_veff(self, mol=None, dm=None, dm_last=0, vhf_last=0, hermi=1):
        'get the HF effective potential for electrons in NEO (rewrite the get_veff method of the SCF class). Be carefule with the effects of :attr:`SCF.direct_scf` on this function'
        #mol = self.mol

        vjk = scf.hf.get_veff(mol, dm, dm_last, vhf_last, hermi)

        #if self.dm_nuc is not None:
        #    jcross = scf.jk.get_jk((mol.elec, mol.elec, mol.nuc, mol.nuc), self.dm_nuc, scripts='ijkl,lk->ij', aosym ='s4')
            #print numpy.einsum('ij,ij',jcross,dm)
        #else:
        #    jcross = 0

        return vjk - self.elec_nuc_coulomb(dm, self.dm_nuc)


    def get_veff_elec(self, dm_elec, dm_nuc):
        'get the HF effective potential for electrons in NEO for given density matrixes of electrons and quantum nuclei'
        mol = self.mol
        vj, vk = scf.jk.get_jk(mol.elec, (dm_elec,dm_elec), ('ijkl,ji->kl','ijkl,jk->il'), aosym='s8')
        jcross = scf.jk.get_jk((mol.elec, mol.elec, mol.nuc, mol.nuc), dm_nuc, scripts='ijkl,lk->ij', aosym ='s4')

        return vj - vk * .5 - jcross

    def get_veff_nuc(self, dm_elec, dm_nuc):
        'get the HF effective potential for quantum nuclei in NEO for given density matrixes of electrons and quantum nuclei'

        mol = self.mol
        vj, vk = scf.jk.get_jk(mol.nuc, (dm_nuc,dm_nuc), ('ijkl,ji->kl','ijkl,jk->il'), aosym='s8')
        jcross = scf.jk.get_jk((mol.nuc, mol.nuc, mol.elec, mol.elec), dm_elec, scripts='ijkl,lk->ij', aosym = 's4')

        if mol.nuc_num == 1 and self.direct == True:
            return -jcross
        else:
            return vj - vk - jcross #still problematic for convergence

    def elec_nuc_coulomb(self, dm_elec, dm_nuc):
        'get the Coulomb energy between electrons and quantum nuclei'
        mol = self.mol
        jcross = scf.jk.get_jk((mol.elec, mol.elec, mol.nuc, mol.nuc), dm_nuc, scripts='ijkl,lk->ij', aosym = 's4')
        return jcross
        #return numpy.einsum('ij,ij', jcross, dm_elec)

    def energy_tot(self, dm_elec, dm_nuc):
        'Total HF energy of NEO'
        mol = self.mol

        energy_classical_nuc = mol.elec.energy_nuc()

        h1e = scf.hf.get_hcore(mol.elec)
        E1_elec = numpy.einsum('ij,ji', h1e, dm_elec)
        vhf_elec = self.get_veff_elec(dm_elec, dm_nuc)
        E_coul_elec = numpy.einsum('ij,ji', vhf_elec, dm_elec) * 0.5

        h1n = self.get_hcore_nuc()
        E1_nuc = numpy.einsum('ij,ji', h1n, dm_nuc)
        vhf_nuc = self.get_veff_nuc(dm_elec, dm_nuc)
        E_coul_nuc = numpy.einsum('ij,ji', vhf_nuc, dm_nuc)* 0.5

        print energy_classical_nuc, E1_elec, E_coul_elec, E1_nuc, E_coul_nuc

        E_tot = energy_classical_nuc  + E1_elec + E_coul_elec  + E1_nuc + E_coul_nuc 
        return E_tot

    def scf_test(self, conv_tot = 1e-7):
        mol = self.mol

        self.dm_elec = scf.hf.init_guess_by_atom(mol.elec)
        self.dm_nuc = self.init_guess_by_core_hamiltonian()

        mf_elec = scf.RHF(mol.elec)
        mf_elec.init_guess = 'atom'
        mf_elec.get_veff = self.get_veff


        #h1n = self.get_hcore_nuc()
        #s1n = scf.hf.get_ovlp(mol.nuc)
        #self.max_cycle = 100

        mf_elec.kernel()

        #self.dm_elec = scf.hf.make_rdm1(mf_elec.mo_coeff, mf_elec.mo_occ)

        h1n = self.get_hcore_nuc()
        s1n = scf.hf.get_ovlp(mol.nuc)
        vhf_nuc = self.get_veff_nuc(self.dm_elec, self.dm_nuc)
        fock_nuc = h1n + vhf_nuc
        no_energy, no_coeff = scf.hf.eig(fock_nuc, s1n)
        no_occ = numpy.zeros(len(no_energy))
        no_occ[:mol.nuc_num] = 1 #high-spin quantum nuclei 
        self.dm_nuc = scf.hf.make_rdm1(no_coeff, no_occ)

        #mf_elec.kernel()

        scf_conv = False
        cycle = 0


    def scf(self, conv_tot = 1e-7):
        'self-consistent field'
        mol = self.mol

        dm_elec = scf.hf.init_guess_by_atom(mol.elec)
        dm_nuc = self.init_guess_by_core_hamiltonian()

        h1e = scf.hf.get_hcore(mol.elec)
        s1e = scf.hf.get_ovlp(mol.elec)

        h1n = self.get_hcore_nuc()
        s1n = scf.hf.get_ovlp(mol.nuc)

        vhf_elec = self.get_veff_elec(dm_elec, dm_nuc)
        vhf_nuc = self.get_veff_nuc(dm_elec, dm_nuc)

        E_tot =  self.energy_tot(dm_elec, dm_nuc)
        print 'Initial energy:',E_tot
        cycle = 0
        scf_conv = False
        max_cycle = 100

        while not scf_conv and cycle <= max_cycle:
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

            vhf_elec = self.get_veff_elec(dm_elec, dm_nuc)

            E_tot = self.energy_tot(dm_elec, dm_nuc)
            print 'Cycle',cycle
            print E_tot

            if abs(E_tot - E_last) < conv_tot:
                scf_conv = True
                print 'Converged'
