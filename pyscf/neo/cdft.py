#!/usr/bin/env python

import sys, numpy, scipy, copy
from pyscf import scf
from pyscf import gto
from pyscf import dft 
from pyscf.neo.rks import KS
from pyscf.lib import logger

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
        self.mol = mol
        self.mf_nuc.get_hcore = self.get_hcore_nuc
        self.scf = self.inner_scf

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

    def Lagrange(self, mf_elec, mf_nuc):
        mol = self.mol
        L = self.energy_tot(mf_elec, mf_nuc) 
        hr = numpy.einsum('xij,x->ij', mol.nuc.intor_symmetric('int1e_r', comp=3), self.f)
        return L + numpy.einsum('ij,ji', hr, dm_nuc)


    def L_first_order(self, f):
        'The first order derivative of L w.r.t the Lagrange multiplier f'
        mol = self.mol
        self.f = f 
        fock = self.mf_nuc.get_fock(dm = self.dm_nuc)
        s1n = self.mf_nuc.get_ovlp()
        energy, coeff = self.mf_nuc.eig(fock, s1n)
        occ = self.mf_nuc.get_occ(energy, coeff)
        self.dm_nuc = scf.hf.make_rdm1(coeff, occ)
        first_order = numpy.einsum('i,xij,j->x', coeff[:,0].conj(), mol.nuc.intor_symmetric('int1e_r', comp=3), coeff[:,0]) - mol.nuclei_expect_position 

        return first_order

    def L_second_order(self):
        'The second order derivative of L w.r.t the Lagrange multiplier f'
        mol = self.mol

        energy = self.mf_nuc.mo_energy
        coeff = self.mf_nuc.mo_coeff

        ints = mol.nuc.intor_symmetric('int1e_r', comp=3)

        de = 1.0/(energy[0] - energy[1:])

        ints = numpy.einsum('...pq,p,qj->...j', ints, coeff[:,0].conj(), coeff[:,1:])
        return 2*numpy.einsum('ij,lj,j->il', ints, ints.conj(), de).real

    def energy_tot(self, mf_elec, mf_nuc):
        'Total energy for cNEO'
        mol = self.mol

        dm_elec = mf_elec.make_rdm1(mf_elec.mo_coeff, mf_elec.mo_occ)
        dm_nuc = mf_nuc.make_rdm1(mf_nuc.mo_coeff, mf_nuc.mo_occ)

        jcross = scf.jk.get_jk((mol.elec, mol.elec, mol.nuc, mol.nuc), dm_nuc, scripts='ijkl,lk->ij', aosym = 's4')
        E_cross = numpy.einsum('ij,ij', jcross, dm_elec)

        hr = numpy.einsum('xij,x->ij', mol.nuc.intor_symmetric('int1e_r', comp=3), self.f)

        E_tot = mf_elec.e_tot + mf_nuc.e_tot - mf_nuc.energy_nuc() + E_cross - numpy.einsum('ij,ji', hr, dm_nuc)

        return E_tot

    def outer_scf(self, conv_tol = 1e-10, max_cycle = 60, method = 1):
        'Outer loop for the optimzation of Lagrange multiplier'

        mol = self.mol
        self.scf()
        f = copy.copy(self.f)
        coeff = self.mf_nuc.mo_coeff
        first_order = numpy.einsum('i,xij,j->x', coeff[:,0].conj(), mol.nuc.intor_symmetric('int1e_r', comp=3), coeff[:,0]) - mol.nuclei_expect_position

        cycle = 0
        conv = False 

        f_set = []
        first_order_set = []

        while numpy.linalg.norm(first_order) > conv_tol:
            f_set.append(copy.copy(f))
            first_order_set.append(first_order)
            cycle += 1
            
            if cycle > max_cycle:
                raise RuntimeError('Not convergent for the optimization of f in %i cycles' %(max_cycle))
            elif cycle <= 2:
                gamma = 1
            else:
                gamma = numpy.dot(f_set[cycle-2] - f_set[cycle-3], first_order_set[cycle-2] - first_order_set[cycle-3])
                gamma = numpy.abs(gamma)/(numpy.linalg.norm(first_order_set[cycle-2] - first_order_set[cycle-3])**2)
            #second_order = self.L_second_order()
            #self.f -= 0.5*numpy.dot(numpy.linalg.inv(second_order), first_order)
            print 'gamma:', gamma
            f += gamma*first_order 
            E_tot = self.scf()
            coeff = self.mf_nuc.mo_coeff
            first_order = numpy.einsum('i,xij,j->x', coeff[:,0].conj(), mol.nuc.intor_symmetric('int1e_r', comp=3), coeff[:,0]) - mol.nuclei_expect_position
            print '1st:', first_order 
            print 'f_set', f_set
            print 'first_order_set', first_order_set

        else:
            print 'Norm of 1st de:', numpy.linalg.norm(first_order)
            print 'f:', f
            return E_tot 

    def optimal_f(self, mf, conv_tol = 1e-10, max_cycle = 50, method = 2):
        'optimization of f'
        mol = self.mol

        first_order = self.L_first_order(self.f)
        cycle = 0

        while numpy.linalg.norm(first_order) > conv_tol:
            cycle += 1
            if method == 1:
                gamma = 1.0/cycle #test
                self.f += gamma*first_order 
            elif method == 2:
                second_order = self.L_second_order()
                #print '2nd:', second_order
                self.f -= 0.5*numpy.dot(numpy.linalg.inv(second_order), first_order) #test
            else:
                raise ValueError('Unsupported method for optimization of f.')

            first_order = self.L_first_order(self.f)
            print 'f:', self.f
            print '1st:', first_order
            if cycle >= max_cycle:
                print 'Error: NOT convergent for the optimation of f.'
                sys.exit(1)
        else:
            print 'Norm of 1st de:', numpy.linalg.norm(first_order)
            print 'f:', self.f
            return self.f


    def inner_scf(self, conv_tol = 1e-8, max_cycle = 100, **kwargs):
        'the self-consistent field driver for the constrained DFT equation of quantum nuclei; Only works for single proton now'

        #self.dm_elec = None
        #self.dm_nuc = None

        '''
        self.mf_elec = dft.RKS(self.mol.elec)
        self.mf_elec.init_guess = 'atom'
        self.mf_elec.xc = 'b3lyp'
        self.mf_elec.grids.level = 5
        self.mf_elec.get_hcore = self.get_hcore_elec

        self.mf_nuc = scf.RHF(self.mol.nuc) #beta: for single proton
        self.mf_nuc.get_init_guess = self.get_init_guess_nuc
        self.mf_nuc.get_hcore = self.get_hcore_nuc
        self.mf_nuc.get_veff = self.get_veff_nuc_bare
        self.mf_nuc.get_occ = self.get_occ_nuc
        '''

        self.mf_elec.kernel()
        self.dm_elec = self.mf_elec.make_rdm1(self.mf_elec.mo_coeff, self.mf_elec.mo_occ)

        self.mf_nuc.kernel()
        self.dm_nuc = self.mf_nuc.make_rdm1(self.mf_nuc.mo_coeff, self.mf_nuc.mo_occ)

        E_tot = self.energy_tot(self.mf_elec, self.mf_nuc)

        self.converged = False
        cycle = 0

        while not self.converged and cycle < max_cycle:
            cycle += 1
            E_last = E_tot

            if cycle >= 1: #using pre-converged density can be more stable 
                #self.f = self.optimal_f(self.mf_nuc)
                opt = scipy.optimize.root(self.L_first_order, self.f, method='hybr')
                self.f = opt.x
                logger.info(self, 'f: %s', self.f)
                logger.info(self, '1st de of L: %s', opt.fun)

            #self.dm_nuc = self.mf_nuc.make_rdm1(self.mf_nuc.mo_coeff, self.mf_nuc.mo_occ)
            self.mf_elec.kernel()
            self.dm_elec = self.mf_elec.make_rdm1(self.mf_elec.mo_coeff, self.mf_elec.mo_occ)

            self.mf_nuc.kernel()
            self.dm_nuc = self.mf_nuc.make_rdm1(self.mf_nuc.mo_coeff, self.mf_nuc.mo_occ)
            E_tot = self.energy_tot(self.mf_elec, self.mf_nuc)
            #first_order = self.L_first_order(self.f)

            logger.info(self, 'Cycle %i Total energy of cNEO: %.15g' %(cycle, E_tot))

            if abs(E_tot - E_last) < conv_tol:
                self.converged = True
                logger.note(self, 'converged cNEO energy = %.15g', E_tot)
                position = numpy.einsum('xij,ji->x', self.mol.nuc.intor_symmetric('int1e_r', comp=3), self.dm_nuc)
                logger.info(self, 'Positional expectation value: %s', position)
                return E_tot

    def nuc_grad_method(self):
        from pyscf.neo.grad import Gradients
        return Gradients(self)
