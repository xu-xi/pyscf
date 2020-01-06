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
    >>> mol.set_quantum_nuclei([0])
    >>> mf = neo.HF(mol)
    >>> mf.scf()
    
    '''

    def __init__(self, mol):
        SCF.__init__(self, mol)

        self.mol = mol
        self.dm_elec = None
        self.dm_nuc = None

        self.mf_elec = scf.RHF(self.mol.elec)
        self.mf_elec.init_guess = 'atom'
        self.mf_elec.get_hcore = self.get_hcore_elec

        self.mf_nuc = scf.RHF(self.mol.nuc) #beta: for single proton
        self.mf_nuc.get_init_guess = self.get_init_guess_nuc
        self.mf_nuc.get_hcore = self.get_hcore_nuc
        self.mf_nuc.get_veff = self.get_veff_nuc_bare
        self.mf_nuc.get_occ = self.get_occ_nuc

        self.verbose = 4

    def get_hcore_nuc(self, mol=None):
        'get core Hamiltonian for quantum nuclei'
        #Z = mol._atm[:,0] # nuclear charge
        #M = gto.mole.atom_mass_list(mol)*1836 # Note: proton mass
        if mol == None:
            mol = self.mol.nuc

        mass_proton = 1836.15267343
        h = mol.intor_symmetric('int1e_kin')/mass_proton
        h -= mol.intor_symmetric('int1e_nuc')
        if self.dm_elec is not None:
            h -= scf.jk.get_jk((mol, mol, self.mol.elec, self.mol.elec), self.dm_elec, scripts='ijkl,lk->ij', aosym ='s4')
        return h

    def get_occ_nuc(self, nuc_energy=None, nuc_coeff=None):
        'label the occupation for protons (high-spin)'

        e_idx = numpy.argsort(nuc_energy)
        e_sort = nuc_energy[e_idx]
        nuc_occ = numpy.zeros(nuc_energy.size)
        nocc = self.mol.nuc_num
        nuc_occ[e_idx[:nocc]] = 1

        return nuc_occ

    def get_init_guess_nuc(self, mol=None, key=None):
        '''Generate initial guess density matrix for quantum nuclei from core hamiltonian

           Returns:
            Density matrix, 2D ndarray
        '''
        if mol == None:
            mol = self.mol.nuc
        h1n = self.get_hcore_nuc(mol)
        s1n = scf.hf.get_ovlp(mol)
        nuc_energy, nuc_coeff = scf.hf.eig(h1n, s1n)
        nuc_occ = self.get_occ_nuc(nuc_energy, nuc_coeff)

        return scf.hf.make_rdm1(nuc_coeff, nuc_occ)
    
    def get_hcore_elec(self, mol=None):
        'get the matrix of core Hamiltonian of electrons of NEO'
        if mol == None:
            mol = self.mol.elec

        if self.dm_nuc is not None:
            jcross = scf.jk.get_jk((mol, mol, self.mol.nuc, self.mol.nuc), self.dm_nuc, scripts='ijkl,lk->ij', aosym ='s4')
        else:
            jcross = 0

        return scf.hf.get_hcore(mol) - jcross

    def get_veff_elec(self, dm_elec, dm_nuc):
        'get the HF effective potential for electrons in NEO for given density matrixes of electrons and quantum nuclei'
        mol = self.mol
        vj, vk = scf.jk.get_jk(mol.elec, (dm_elec,dm_elec), ('ijkl,ji->kl','ijkl,jk->il'), aosym='s8')
        jcross = scf.jk.get_jk((mol.elec, mol.elec, mol.nuc, mol.nuc), dm_nuc, scripts='ijkl,lk->ij', aosym ='s4')

        return vj - vk * .5 - jcross

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
        'get the Coulomb matrix between electrons and quantum nuclei'
        mol = self.mol
        jcross = scf.jk.get_jk((mol.elec, mol.elec, mol.nuc, mol.nuc), dm_nuc, scripts='ijkl,lk->ij', aosym = 's4')
        return jcross
        #return numpy.einsum('ij,ij', jcross, dm_elec)

    def energy_tot(self, mf_elec, mf_nuc):
        'Total energy of NEO'
        mol = self.mol
        
        dm_elec = scf.hf.make_rdm1(mf_elec.mo_coeff, mf_elec.mo_occ)
        dm_nuc = scf.hf.make_rdm1(mf_nuc.mo_coeff, mf_nuc.mo_occ)

        jcross = scf.jk.get_jk((mol.elec, mol.elec, mol.nuc, mol.nuc), dm_nuc, scripts='ijkl,lk->ij', aosym = 's4')
        E_cross = numpy.einsum('ij,ij', jcross, dm_elec)

        E_tot = mf_elec.e_tot + mf_nuc.e_tot - mf_nuc.energy_nuc() + E_cross

        logger.debug(self, 'Energy of electrons: %s',  mf_elec.e_tot)
        logger.debug(self, 'Energy of quantum nuclei: %s', mf_nuc.e_tot)
        logger.debug(self, 'Energy of classcial nuclei: %s', mf_nuc.energy_nuc())
        logger.debug(self, 'Energy of e-p Comlomb interaction: %s', E_cross)

        return E_tot

    def scf(self, conv_tol = 1e-7, max_cycle = 60, dm0_elec = None, dm0_nuc = None):
        'self-consistent field driver for NEO'
        mol = self.mol

        self.mf_elec.kernel()
        self.dm_elec = scf.hf.make_rdm1(self.mf_elec.mo_coeff, self.mf_elec.mo_occ)

        self.mf_nuc.kernel()

        E_tot = self.energy_tot(self.mf_elec, self.mf_nuc)

        scf_conv = False
        cycle = 0

        while not scf_conv:
            cycle += 1
            if cycle > max_cycle:
                raise RuntimeError('SCF is not convergent within %i cycles' %(max_cycle))

            E_last = E_tot
            self.dm_nuc = scf.hf.make_rdm1(self.mf_nuc.mo_coeff, self.mf_nuc.mo_occ)
            self.mf_elec.kernel()
            self.dm_elec = scf.hf.make_rdm1(self.mf_elec.mo_coeff, self.mf_elec.mo_occ)
            self.mf_nuc.kernel()
            E_tot = self.energy_tot(self.mf_elec, self.mf_nuc)
            logger.info(self, 'Cycle %i Total Energy of NEO: %s' %(cycle, E_tot))
            if abs(E_tot - E_last) < conv_tol:
                scf_conv = True
                logger.note(self, 'converged NEO energy = %.15g', E_tot) 
                return E_tot
