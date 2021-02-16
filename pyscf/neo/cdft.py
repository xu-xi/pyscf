#!/usr/bin/env python

'''
Constrained nuclear-electronic orbital density functional theory
'''
import sys, numpy, scipy, copy
from pyscf import scf
from pyscf import gto
from pyscf import dft 
from pyscf.neo.ks import KS
from pyscf.lib import logger
from pyscf.data import nist

class CDFT(KS):
    '''
    Example:

    >>> mol = neo.Mole()
    >>> mol.build(atom = 'H 0.0 0.0 0.0; C 0.0 0.0 1.064; N 0.0 0.0 2.220', basis = 'ccpvdz')
    >>> mf = neo.CDFT(mol)
    >>> mf.scf()
    '''

    def __init__(self, mol):
        KS.__init__(self, mol)
        self.scf = self.inner_scf
        self.f = [numpy.zeros(3)] * self.mol.natm

    def build(self):
        KS.build(self)

        # set up the Hamiltonian for each quantum nuclei in cNEO
        for i in range(len(self.mol.nuc)):
            mf = self.mf_nuc[i]
            mf.nuclei_expect_position = mf.mol.atom_coord(mf.mol.atom_index)
            mf.get_hcore = self.get_hcore_nuc
            #self.dm_nuc[i] = self.get_init_guess_nuc(self.mol.nuc[i])

    def get_hcore_nuc(self, mol):
        'get the core Hamiltonian for quantum nucleus in cNEO'
        ia = mol.atom_index
        mass = self.mol.mass[ia] * nist.ATOMIC_MASS/nist.E_MASS # the mass of quantum nucleus in a.u.
        charge = self.mol.atom_charge(ia)

        h = mol.intor_symmetric('int1e_kin')/mass
        h -= mol.intor_symmetric('int1e_nuc')*charge

        # Coulomb interactions between quantum nucleus and electrons
        if self.unrestricted == True:
            h -= scf.jk.get_jk((mol, mol, self.mol.elec, self.mol.elec), self.dm_elec[0], scripts='ijkl,lk->ij', aosym ='s4')*charge
            h -= scf.jk.get_jk((mol, mol, self.mol.elec, self.mol.elec), self.dm_elec[1], scripts='ijkl,lk->ij', aosym ='s4')*charge
        else:
            h -= scf.jk.get_jk((mol, mol, self.mol.elec, self.mol.elec), self.dm_elec, scripts='ijkl,lk->ij', intor='int2e', aosym ='s4')*charge

        # Coulomb interactions between quantum nuclei
        for j in range(len(self.dm_nuc)):
            ja = self.mol.nuc[j].atom_index
            if ja != ia and isinstance(self.dm_nuc[j], numpy.ndarray):
                h += scf.jk.get_jk((mol, mol, self.mol.nuc[j], self.mol.nuc[j]), self.dm_nuc[j], scripts='ijkl,lk->ij')*charge*self.mol.atom_charge(ja) # times nuclear charge

        # extra term in cNEO due to the constraint on expectational position
        h += numpy.einsum('xij,x->ij', mol.intor_symmetric('int1e_r', comp=3), self.f[ia])

        return h

    def Lagrange(self, mf_elec, mf_nuc):
        mol = self.mol
        L = self.energy_tot(mf_elec, mf_nuc) 
        hr = numpy.einsum('xij,x->ij', mol.nuc.intor_symmetric('int1e_r', comp=3), self.f)
        return L + numpy.einsum('ij,ji', hr, dm_nuc)

    def first_order_de(self, f, mf):
        'The first order derivative of L w.r.t the Lagrange multiplier f'
        mol = self.mol
        index = self.mf_nuc.index(mf)
        i = mf.mol.atom_index
        self.f[i] = f
        #fock = mf.get_fock(dm = self.dm_nuc)
        h1n = mf.get_hcore(mol = mf.mol)
        s1n = mf.get_ovlp()
        energy, coeff = mf.eig(h1n, s1n)
        occ = mf.get_occ(energy, coeff)
        self.dm_nuc[index] = scf.hf.make_rdm1(coeff, occ)

        return numpy.einsum('i,xij,j->x', coeff[:,0].conj(), mf.mol.intor_symmetric('int1e_r', comp=3), coeff[:,0]) - mf.nuclei_expect_position


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
        E_tot = 0 

        self.dm_elec = mf_elec.make_rdm1()
        for i in range(len(mf_nuc)):
            self.dm_nuc[i] = mf_nuc[i].make_rdm1()

        h1e = mf_elec.get_hcore(mf_elec.mol)
        vhf = mf_elec.get_veff(mf_elec.mol, self.dm_elec)
        E_tot += mf_elec.energy_elec(dm = self.dm_elec, h1e = h1e, vhf = vhf)[0] 

        for i in range(len(mf_nuc)):
            ia = mf_nuc[i].mol.atom_index
            h1n = mf_nuc[i].get_hcore(mf_nuc[i].mol)
            n1 = numpy.einsum('ij,ji', h1n, self.dm_nuc[i])
            logger.debug(self, 'Energy of %s: %s', self.mol.atom_symbol(ia), n1)
            E_tot += n1
            h_r = numpy.einsum('xij,x->ij', mf_nuc[i].mol.intor_symmetric('int1e_r', comp=3), self.f[ia])
            E_tot -= numpy.einsum('ij,ji', h_r, self.dm_nuc[i])

            if self.mol.atom_symbol(ia) == 'H' and self.epc is not None:
                veff = self.mf_nuc[i].get_veff(self.mf_nuc[i].mol, self.dm_nuc[i])
                E_tot += veff.exc

        E_tot = E_tot - self.elec_nuc_coulomb(self.dm_elec, self.dm_nuc) - self.nuc_nuc_coulomb(self.dm_nuc) + mf_elec.energy_nuc() # substract repeatedly counted terms

        return E_tot

    def outer_scf(self, conv_tol = 1e-10, max_cycle = 60, method = 1):
        'Outer loop for the optimzation of Lagrange multiplier'

        mol = self.mol
        self.scf()
        f = copy.copy(self.f)
        coeff = self.mf_nuc.mo_coeff
        first_order = numpy.einsum('i,xij,j->x', coeff[:,0].conj(), mol.nuc.intor_symmetric('int1e_r', comp=3), coeff[:,0]) - self.nuclei_expect_position

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
            logger.info(self, 'gamma: %s', gamma)
            f += gamma*first_order 
            E_tot = self.scf()
            coeff = self.mf_nuc.mo_coeff
            first_order = numpy.einsum('i,xij,j->x', coeff[:,0].conj(), mol.nuc.intor_symmetric('int1e_r', comp=3), coeff[:,0]) - self.nuclei_expect_position
            logger.info(self, '1st de:%s', first_order)

        else:
            logger.info(self, 'Norm of 1st de: %s', numpy.linalg.norm(first_order))
            logger.info(self, 'f:', f)
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
                self.f -= 0.5*numpy.dot(numpy.linalg.inv(second_order), first_order) #test
            else:
                raise ValueError('Unsupported method for optimization of f.')

            first_order = self.L_first_order(self.f)
            logger.info(self, 'f:%s', self.f)
            logger.info(self, '1st de: %s', first_order)
            if cycle >= max_cycle:
                raise RuntimeError('NOT convergent for the optimation of f.')
        else:
            logger(self, 'Norm of 1st de: %s', numpy.linalg.norm(first_order))
            logger(self, 'f: %s', self.f)
            return self.f


    def inner_scf(self, conv_tol = 1e-8, max_cycle = 20, opt_method = 'hybr', **kwargs):
        'the self-consistent field driver for the constrained DFT equation of quantum nuclei'
        
        self.build()

        self.mf_elec.kernel(self.dm_elec, dump_chk=None)
        self.dm_elec = self.mf_elec.make_rdm1()

        if self.unrestricted == True: # use stability analysis to make initial electronic density matrix
            mo = self.mf_elec.stability()[0]
            self.dm_elec = self.mf_elec.make_rdm1(mo, self.mf_elec.mo_occ)
            self.mf_elec.max_cycle = 200

        for i in range(len(self.mol.nuc)):
            self.dm_nuc[i] = self.get_init_guess_nuc(self.mf_nuc[i])
            self.mf_nuc[i].kernel(self.dm_nuc[i], dump_chk=None)
            self.dm_nuc[i] = self.mf_nuc[i].make_rdm1()

        E_tot = self.energy_tot(self.mf_elec, self.mf_nuc)
        logger.info(self, 'Initial total Energy of NEO: %.15g\n' %(E_tot))

        cycle = 0

        while not self.converged:
            cycle += 1
            if cycle > max_cycle:
                raise RuntimeError('SCF is not convergent within %i cycles' %(max_cycle))

            E_last = E_tot

            self.mf_elec.kernel(self.dm_elec, dump_chk=None)
            self.dm_elec = self.mf_elec.make_rdm1()
            if self.unrestricted == True:
                mo = self.mf_elec.stability()[0]
                self.dm_elec = self.mf_elec.make_rdm1(mo, self.mf_elec.mo_occ)
                self.mf_elec.max_cycle = 200

            #if cycle >= 1: # using pre-converged density can be more stable
            for i in range(len(self.mf_nuc)):
                mf = self.mf_nuc[i]
                index = mf.mol.atom_index
                #self.f = self.optimal_f(self.mf_nuc)
                opt = scipy.optimize.root(self.first_order_de, self.f[index], args=mf, method=opt_method)
                self.f[index] = opt.x
                logger.info(self, 'f of %s(%i) atom: %s' %(self.mol.atom_symbol(index), index, self.f[index]))
                logger.info(self, '1st de of L: %s', opt.fun)
                self.mf_nuc[i].kernel(self.dm_nuc[i], dump_chk=None)
                self.dm_nuc[i] = self.mf_nuc[i].make_rdm1()

            E_tot = self.energy_tot(self.mf_elec, self.mf_nuc)
            logger.info(self, 'Cycle %i Total energy of cNEO: %.15g\n' %(cycle, E_tot))

            if abs(E_tot - E_last) < conv_tol:
                self.converged = True
                logger.note(self, 'converged cNEO energy = %.15g', E_tot)
                for i in range(len(self.mf_nuc)):
                    position = numpy.einsum('xij,ji->x', self.mol.nuc[i].intor_symmetric('int1e_r', comp=3), self.dm_nuc[i])
                    logger.info(self, 'Positional expectation value of the %i-th atom: %s', self.mol.nuc[i].atom_index, position)
                return E_tot

    def nuc_grad_method(self):
        from pyscf.neo.grad import Gradients
        return Gradients(self)
