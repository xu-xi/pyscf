#!/usr/bin/env python

import numpy, sys
from pyscf import scf
from pyscf import gto
from pyscf.neo.rks import KS
from pyscf.tdscf.rhf import _charge_center

class CDFT(KS):
    '''
    Example:

    >>> mol = neo.Mole()
    >>> mol.build(atom = 'H 0.0 0.0 0.0; C 0.0 0.0 1.064; N 0.0 0.0 2.220', basis = 'ccpvdz')
    >>> mol.set_quantum_nuclei([0])
    >>> mol.set_nuclei_expect_position(mol.atom_coord(0), unit='B')
    >>> mf = neo.CDFT(mol)
    >>> mf.mf_elec.xc = 'b3lyp'
    >>> mf.scf()
    '''

    def __init__(self, mol):
        KS.__init__(self, mol)

        self.f = numpy.zeros(3) 
        self.mf_nuc.get_hcore = self.get_hcore_nuc

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

        h += numpy.einsum('xij,x->ij', mol.intor_symmetric('int1e_r', comp=3), self.f)

        return h

    def L_second_order(self, energy, coeff):
        'calculate the second order of L w.r.t the lagrange multiplier f'
        mol = self.mol
        ints = mol.nuc.intor_symmetric('int1e_r', comp=3)

        de = 1.0/(energy[0] - energy[1:])

        ints = numpy.einsum('...pq,p,qj->...j', ints, coeff[:,0].conj(), coeff[:,1:])
        return 2*numpy.einsum('ij,lj,j->il', ints, ints.conj(), de).real

    def energy_tot(self, mf_elec, mf_nuc):
        'Total energy for cNEO'
        mol = self.mol

        dm_elec = scf.hf.make_rdm1(mf_elec.mo_coeff, mf_elec.mo_occ)
        dm_nuc = scf.hf.make_rdm1(mf_nuc.mo_coeff, mf_nuc.mo_occ)

        jcross = scf.jk.get_jk((mol.elec, mol.elec, mol.nuc, mol.nuc), dm_nuc, scripts='ijkl,lk->ij', aosym = 's4')
        E_cross = numpy.einsum('ij,ij', jcross, dm_elec)

        hr = numpy.einsum('xij,x->ij', mol.nuc.intor_symmetric('int1e_r', comp=3), self.f)
        E_tot = mf_elec.e_tot + mf_nuc.e_tot - mf_nuc.energy_nuc() + E_cross - numpy.einsum('ij,ji', hr, dm_nuc)

        return E_tot

    def outer_scf(self, conv_tol = 1e-10, max_cycle = 100):
        'Outer loop for the optimzation of Lagrange multiplier'

        mol = self.mol
        self.scf()

        cycle = 0
        conv = False 

        while not conv and cycle < max_cycle:
            cycle += 1
            
            #print 'Energy:', self.mf_nuc.mo_energy
            first_order = numpy.einsum('i,xij,j->x', self.mf_nuc.mo_coeff[:,0].conj(), mol.nuc.intor_symmetric('int1e_r', comp=3), self.mf_nuc.mo_coeff[:,0]) - mol.nuclei_expect_position 
            print '1st:', first_order 
            if numpy.linalg.norm(first_order) < conv_tol:
                conv = True
            else:
                second_order = self.L_second_order(self.mf_nuc.mo_energy, self.mf_nuc.mo_coeff)
                print '2nd:', second_order
                self.f -= numpy.dot(numpy.linalg.inv(second_order), first_order)
                self.scf()

        if conv:
            print 'Norm of 1st de:', numpy.linalg.norm(first_order)
            print 'f:', self.f
            return self.f
        else:
            print 'Error: NOT convergent'
            sys.exit(1)

    def newton_opt(self, mf, conv_tol = 1e-10, max_cycle = 50):
        'Newton optimization'
        mol = self.mol

        s1n = mf.get_ovlp()
        energy = mf.mo_energy
        coeff = mf.mo_coeff

        cycle =0
        conv = False

        while not conv and cycle < max_cycle:
            cycle += 1
            first_order = numpy.einsum('i,xij,j->x', coeff[:,0].conj(), mol.nuc.intor_symmetric('int1e_r', comp=3), coeff[:,0]) - mol.nuclei_expect_position

            if numpy.linalg.norm(first_order) < conv_tol:
                conv = True
            else:
                second_order = self.L_second_order(energy, coeff)
                self.f -= numpy.dot(numpy.linalg.inv(second_order), first_order)
                fock = mf.get_fock(dm = self.dm_nuc)
                energy, coeff = mf.eig(fock, s1n)
                occ = mf.get_occ(energy, coeff)
                self.dm_nuc = scf.hf.make_rdm1(coeff, occ)

        if conv:
            print 'Norm of 1st de:', numpy.linalg.norm(first_order)
            print 'f:', self.f
            return self.f
        else:
            print 'Error: NOT convergent for the optimation of f.'
            sys.exit(1)

    def inner_scf(self, conv_tol = 1e-7, max_cycle = 100):
        'the self-consistent field driver for the constrained DFT equation of quantum nuclei; Only works for single proton now'
        mol = self.mol

        self.mf_elec.kernel()
        self.dm_elec = scf.hf.make_rdm1(self.mf_elec.mo_coeff, self.mf_elec.mo_occ)

        self.mf_nuc.kernel()
        self.dm_nuc = scf.hf.make_rdm1(self.mf_nuc.mo_coeff, self.mf_nuc.mo_occ)

        E_tot = self.energy_tot(self.mf_elec, self.mf_nuc)

        scf_conv = False
        cycle = 0

        while not scf_conv and cycle <= max_cycle:
            cycle += 1
            E_last = E_tot

            if cycle >= 1: #using pre-converged density can be more stable 
                self.f = self.newton_opt(self.mf_nuc)

            self.mf_elec.kernel()
            self.dm_elec = scf.hf.make_rdm1(self.mf_elec.mo_coeff, self.mf_elec.mo_occ)

            self.mf_nuc.kernel()
            self.dm_nuc = scf.hf.make_rdm1(self.mf_nuc.mo_coeff, self.mf_nuc.mo_occ)

            E_tot = self.energy_tot(self.mf_elec, self.mf_nuc)

            print 'Cycle',cycle
            print 'Total energy of cNEO:', E_tot

            if abs(E_tot - E_last) < conv_tol:
                scf_conv = True
                print 'Converged'
                print 'Positional expectation value:', numpy.einsum('xij,ji->x', mol.nuc.intor_symmetric('int1e_r', comp=3), self.dm_nuc)
                return E_tot

