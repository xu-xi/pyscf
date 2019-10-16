#!/usr/bin/env python

import numpy
from pyscf import scf
from pyscf import gto
from pyscf.neo.rks import KS

class CDFT(KS):
    '''
    Example:
    
    '''
    def __init__(self, mol):
        KS.__init__(self, mol)

        self.f = numpy.array([0.0,0.0,0.0],dtype='float64')
        self.mf_nuc = scf.RHF(mol.nuc)
        self.mf_nuc.get_init_guess = self.get_init_guess_nuc
        self.mf_nuc.get_hcore = self.get_hcore_nuc
        self.mf_nuc.get_veff = self.get_veff_nuc
        self.mf_nuc.get_occ = self.get_occ_nuc

    def get_hcore_nuc(self, mol=None):
        'get core Hamiltonian for quantum nuclei'
        #Z = mol._atm[:,0] # nuclear charge
        #M = gto.mole.atom_mass_list(mol)*1836 # Note: proton mass
        if mol == None:
            mol = self.mol.nuc

        mass_proton = 1836.15267343
        h = gto.moleintor.getints('int1e_kin_sph', mol._atm, mol._bas, mol._env, hermi=1, aosym='s4')/mass_proton
        h -= gto.moleintor.getints('int1e_nuc_sph', mol._atm, mol._bas, mol._env, hermi=1, aosym='s4')        
        if self.dm_elec is not None:
            h -= scf.jk.get_jk((mol, mol, self.mol.elec, self.mol.elec), self.dm_elec, scripts='ijkl,lk->ij', aosym ='s4')

        if mol.nuclei_expect_position is not None:
            h += numpy.einsum('xij,x->ij', mol.intor_symmetric('int1e_r', comp=3), self.f)

        return h

    def L_second_order(self, energy, coeff):
        'calculate the second order of L w.r.t the lagrange multiplier f'
        mol = self.mol
        ints = mol.nuc.intor_symmetric('int1e_r', comp=3)

        de = 1.0/(energy[0] - energy[1:])

        ints = numpy.einsum('...pq,p,qj->...j', ints, coeff[:,0].conj(), coeff[:,1:])
        return 2*numpy.einsum('ij,lj,j->il', ints, ints.conj(), de).real

    def newton_opt(self, mf_nuc):

        max_cycle = 10
        mol = self.mol

        fock = self.get_hcore_nuc(mol.nuc)
        s1n = mf_nuc.get_ovlp()
        no_energy, no_coeff = self.mf_nuc.eig(fock, s1n)
        first_order = numpy.einsum('i,xij,j->x', no_coeff[:,0].conj(), mol.nuc.intor_symmetric('int1e_r', comp=3), no_coeff[:,0]) - mol.nuclei_expect_position 
        cycle =0

        while not numpy.linalg.norm(first_order) < 1e-5 and cycle < max_cycle:
            #print 'first order:', first_order
            cycle += 1
            second_order = self.L_second_order(no_energy, no_coeff)
            #print 'Condition number of 2nd de:', numpy.linalg.cond(second_order)
            self.f -= numpy.dot(numpy.linalg.pinv(second_order), first_order)
            fock = self.get_hcore_nuc(self.mol.nuc)
            no_energy, no_coeff = self.mf_nuc.eig(fock, s1n)
            first_order = numpy.einsum('i,xij,j->x', no_coeff[:,0].conj(), mol.nuc.intor_symmetric('int1e_r', comp=3), no_coeff[:,0]) - mol.nuclei_expect_position 
            #print 'Norm of 1st de:', numpy.linalg.norm(first_order)

        print 'Norm of 1st de:', numpy.linalg.norm(first_order)
        print 'f:', self.f
        return self.f

    def scf_cdft_nuc(self, conv_tot=1e-7):
        'the self-consistent field driver for the constrained DFT equation of quantum nuclei'
        max_cycle = 100
        mol = self.mol
        self.mf_elec.kernel() #should delete
        self.dm_elec = scf.hf.make_rdm1(self.mf_elec.mo_coeff, self.mf_elec.mo_occ)

        self.f = self.newton_opt(self.mf_nuc)
        h1n = self.mf_nuc.get_hcore()
        s1n = self.mf_nuc.get_ovlp()
        no_energy, no_coeff = scf.hf.eig(h1n, s1n)
        no_occ = numpy.zeros(len(no_energy))
        no_occ[:mol.nuc_num] = 1 #high-spin quantum nuclei
        self.dm_nuc = scf.hf.make_rdm1(no_coeff, no_occ)

        jcross = scf.jk.get_jk((mol.elec, mol.elec, mol.nuc, mol.nuc), self.dm_nuc, scripts='i    jkl,lk->ij', aosym = 's4')
        E_tot = self.mf_elec.e_tot + numpy.einsum('ij,ji', h1n, self.dm_nuc) + numpy.einsum('ij,ij', jcross, self.dm_elec)

        print 'Initial energy:', E_tot
        scf_conv = False
        cycle = 0

        while not scf_conv and cycle <= max_cycle:
            cycle += 1
            E_last = E_tot

            self.mf_elec.kernel()
            self.dm_elec = scf.hf.make_rdm1(self.mf_elec.mo_coeff, self.mf_elec.mo_occ)

            self.f = self.newton_opt(self.mf_nuc)

            h1n = self.mf_nuc.get_hcore()
            no_energy, no_coeff = scf.hf.eig(h1n, s1n)
            no_occ = numpy.zeros(len(no_energy))
            no_occ[:mol.nuc_num] = 1 #high-spin quantum nuclei
            self.dm_nuc = scf.hf.make_rdm1(no_coeff, no_occ)
            jcross = scf.jk.get_jk((mol.elec, mol.elec, mol.nuc, mol.nuc), self.dm_nuc, scripts='ijkl,lk->ij', aosym = 's4')

            E_tot = self.mf_elec.e_tot + numpy.einsum('ij,ji', h1n, self.dm_nuc) + numpy.einsum('ij,ij', jcross, self.dm_elec)

            print 'Cycle',cycle
            print E_tot

            if abs(E_tot - E_last) < conv_tot:
                scf_conv = True
                print 'Converged'
                return E_tot


    def scf(self, conv_tot=1e-7):
        max_cycle = 10
        mol = self.mol

        self.mf_elec.kernel()
        self.dm_elec = scf.hf.make_rdm1(self.mf_elec.mo_coeff, self.mf_elec.mo_occ)

        self.f = self.newton_opt(self.mf_nuc)
        self.mf_nuc.kernel()
        self.dm_nuc = scf.hf.make_rdm1(self.mf_nuc.mo_coeff, self.mf_nuc.mo_occ)

        scf_conv = False
        cycle =0

          
