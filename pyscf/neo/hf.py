#!/usr/bin/env python

'''
Nuclear Electronic Orbital Hartree-Fock (NEO-HF)
'''

import numpy
from pyscf import gto
from pyscf import scf
from pyscf import lib
from pyscf.lib import logger
from pyscf.scf.hf import SCF


class HF(SCF):
    '''Hartree Fock for NEO
    
    Example:
    
    >>> from pyscf import neo
    >>> mol = neo.Mole()
    >>> mol.build(atom = 'H 0 0 0; F 0 0 0.917', basis = 'ccpvdz')
    >>> mf = neo.HF(mol)
    >>> mf.scf()
    
    '''

    def __init__(self, mol):
        SCF.__init__(self, mol)

        self.mol = mol
        self.dm_elec = None
        self.dm_nuc = [None]*self.mol.nuc_num
        self.verbose = 5

    def get_hcore_nuc(self, nole):
        'get core Hamiltonian for quantum nucleus. Nole is the Mole object for it.'

        i = nole.atom_index
        mass = 1836.15267343 * self.mol.atom_mass_list()[i] # the mass of quantum nucleus in a.u.

        h = nole.intor_symmetric('int1e_kin')/mass
        h -= nole.intor_symmetric('int1e_nuc')*self.mol._atm[i,0] # times nuclear charge

        # Coulomb interactions between quantum nucleus and electrons
        if isinstance(self.dm_elec, numpy.ndarray):
            h -= scf.jk.get_jk((nole, nole, self.mol.elec, self.mol.elec), self.dm_elec, scripts='ijkl,lk->ij', aosym ='s4') * self.mol._atm[i,0]

        # Coulomb interactions between quantum nuclei
        for j in range(len(self.dm_nuc)):
            k = self.mol.nuc[j].atom_index
            if k != i and isinstance(self.dm_nuc[j], numpy.ndarray):
                h += scf.jk.get_jk((nole, nole, self.mol.nuc[j], self.mol.nuc[j]), self.dm_nuc[j], scripts='ijkl,lk->ij') * self.mol._atm[i, 0] * self.mol._atm[k, 0] # times nuclear charge

        return h

    def get_occ_nuc(self, nuc_energy=None, nuc_coeff=None):
        'label the occupation for quantum nucleus'

        e_idx = numpy.argsort(nuc_energy)
        e_sort = nuc_energy[e_idx]
        nuc_occ = numpy.zeros(nuc_energy.size)
        #nocc = self.mol.nuc_num
        nocc = 1
        nuc_occ[e_idx[:nocc]] = 1

        return nuc_occ

    def get_init_guess_nuc(self, nole, key=None):
        '''Generate initial guess density matrix for quantum nuclei from core hamiltonian

           Returns:
            Density matrix, 2D ndarray
        '''
        h1n = self.get_hcore_nuc(nole)
        s1n = nole.intor_symmetric('int1e_ovlp')
        nuc_energy, nuc_coeff = scf.hf.eig(h1n, s1n)
        nuc_occ = self.get_occ_nuc(nuc_energy, nuc_coeff)

        return scf.hf.make_rdm1(nuc_coeff, nuc_occ)
    
    def get_hcore_elec(self, eole=None):
        'Get the core Hamiltonian for electrons in NEO'
        if eole == None:
            eole = self.mol.elec # the Mole object for electrons in NEO

        j = 0
        # Coulomb interactions between electrons and all quantum nuclei
        for i in range(len(self.dm_nuc)):
            if isinstance(self.dm_nuc[i], numpy.ndarray):
                j -= scf.jk.get_jk((eole, eole, self.mol.nuc[i], self.mol.nuc[i]), self.dm_nuc[i], scripts='ijkl,lk->ij', aosym='s4') * self.mol._atm[self.mol.nuc[i].atom_index, 0]

        return scf.hf.get_hcore(eole) + j

    def get_veff_nuc_bare(self, mol, dm, dm_last=None, vhf_last=None, hermi=1, vhfopt=None):
        'NOTE: Only for single quantum proton system.'
        return numpy.zeros((mol.nao_nr(), mol.nao_nr()))

    def get_veff_nuc(self, mol, dm, dm_last=None, vhf_last=None, hermi=1, vhfopt=None):
        'get the HF effective potential for quantum nuclei in NEO'

        if dm_last is None:
            vj, vk = scf.jk.get_jk(mol, (dm, dm), ('ijkl,ji->kl','ijkl,jk->il'), aosym='s8')
            return vj - vk
        else:
            ddm = numpy.asarray(dm) - numpy.asarray(dm_last)
            vj, vk = scf.jk.get_jk(mol, (ddm, ddm), ('ijkl,ji->kl','ijkl,jk->il'), aosym='s8')
            return vj - vk  + numpy.asarray(vhf_last)

    def elec_nuc_coulomb(self, dm_elec, dm_nuc):
        'the energy of Coulomb interactions between electrons and quantum nuclei'
        mol = self.mol
        jcross = 0
        for i in range(len(dm_nuc)):
            jcross -= scf.jk.get_jk((mol.elec, mol.elec, mol.nuc[i], mol.nuc[i]), dm_nuc[i], scripts='ijkl,lk->ij', aosym = 's4') * mol._atm[mol.nuc[i].atom_index, 0]
        E = numpy.einsum('ij,ji', jcross, dm_elec)
        logger.debug(self, 'Energy of e-n Comlomb interactions: %s', E)
        return E

    def nuc_nuc_coulomb(self, dm_nuc):
        'the energy of Coulomb interactions between quantum nuclei'
        mol = self.mol
        E = 0
        for i in range(len(dm_nuc)):
            for j in range(len(dm_nuc)):
                if j != i:
                    jcross = scf.jk.get_jk((mol.nuc[i], mol.nuc[i], mol.nuc[j], mol.nuc[j]), dm_nuc[j], scripts='ijkl,lk->ij', aosym='s4') * mol._atm[mol.nuc[i].atom_index, 0] * mol._atm[mol.nuc[j].atom_index, 0]
                    E += numpy.einsum('ij,ji', jcross, dm_nuc[i])

        logger.debug(self, 'Energy of n-n Comlomb interactions: %s', E*.5) # double counted
        return E*.5 


    def energy_tot(self, mf_elec, mf_nuc):
        'Total energy of NEO'
        mol = self.mol
        
        dm_elec = scf.hf.make_rdm1(mf_elec.mo_coeff, mf_elec.mo_occ)
        dm_nuc = [None]*self.mol.nuc_num
        for i in range(len(mf_nuc)):
            dm_nuc[i] = scf.hf.make_rdm1(mf_nuc[i].mo_coeff, mf_nuc[i].mo_occ)

        E_tot = 0
        for i in range(len(mf_nuc)):
            E_tot += mf_nuc[i].e_tot
        logger.debug(self, 'Energy of quantum nuclei: %s', E_tot)
        logger.debug(self, 'Energy of electrons: %s',  mf_elec.e_tot)
        logger.debug(self, 'Energy of classcial nuclei: %s', mf_elec.energy_nuc())

        E_tot += mf_elec.e_tot - self.elec_nuc_coulomb(dm_elec, dm_nuc) - self.nuc_nuc_coulomb(dm_nuc) - self.mol.nuc_num * mf_elec.energy_nuc() # substract repeatedly counted terms

        return E_tot

    def scf(self, conv_tol = 1e-7, max_cycle = 60, dm0_elec = None, dm0_nuc = None):
        'self-consistent field driver for NEO'

        # set up the Hamiltonian for electrons
        self.mf_elec = scf.RHF(self.mol.elec)
        self.mf_elec.init_guess = 'atom'
        self.mf_elec.get_hcore = self.get_hcore_elec
        self.dm_elec = self.mf_elec.get_init_guess(key='atom')

        # set up the Hamiltonian for each quantum nucleus
        self.mf_nuc = [None] * self.mol.nuc_num
        for i in range(len(self.mol.nuc)):
            self.mf_nuc[i] = scf.RHF(self.mol.nuc[i])
            self.mf_nuc[i].get_init_guess = self.get_init_guess_nuc
            self.mf_nuc[i].get_hcore = self.get_hcore_nuc
            self.mf_nuc[i].get_veff = self.get_veff_nuc_bare
            self.mf_nuc[i].get_occ = self.get_occ_nuc
            self.dm_nuc[i] = self.get_init_guess_nuc(self.mol.nuc[i])

        self.mf_elec.kernel(dump_chk=False)
        self.dm_elec = scf.hf.make_rdm1(self.mf_elec.mo_coeff, self.mf_elec.mo_occ)

        for i in range(len(self.mf_nuc)):
            self.mf_nuc[i].kernel(dump_chk=False)
            self.dm_nuc[i] = scf.hf.make_rdm1(self.mf_nuc[i].mo_coeff, self.mf_nuc[i].mo_occ)

        E_tot = self.energy_tot(self.mf_elec, self.mf_nuc)
        logger.info(self, 'Initial total Energy of NEO: %.15g' %(E_tot))

        scf_conv = False
        cycle = 0

        while not scf_conv:
            cycle += 1
            if cycle > max_cycle:
                raise RuntimeError('SCF is not convergent within %i cycles' %(max_cycle))

            E_last = E_tot
            #self.dm_nuc = scf.hf.make_rdm1(self.mf_nuc.mo_coeff, self.mf_nuc.mo_occ)
            self.mf_elec.kernel(dump_chk=False)
            self.dm_elec = scf.hf.make_rdm1(self.mf_elec.mo_coeff, self.mf_elec.mo_occ)
            for i in range(len(self.mf_nuc)):
                self.mf_nuc[i].kernel(dump_chk=False)
                self.dm_nuc[i] = scf.hf.make_rdm1(self.mf_nuc[i].mo_coeff, self.mf_nuc[i].mo_occ)

            E_tot = self.energy_tot(self.mf_elec, self.mf_nuc)
            logger.info(self, 'Cycle %i Total Energy of NEO: %s' %(cycle, E_tot))
            if abs(E_tot - E_last) < conv_tol:
                print(self.mf_elec.mo_energy)
                print(self.mf_nuc[0].mo_energy)
                scf_conv = True
                logger.note(self, 'converged NEO energy = %.15g', E_tot) 
                return E_tot
