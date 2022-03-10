#!/usr/bin/env python

'''
Nuclear Electronic Orbital Hartree-Fock (NEO-HF) for periodic systems at a single k-point
'''

import numpy
import scipy
from pyscf import scf
from pyscf import neo
from pyscf import pbc
from pyscf.pbc import tools
from pyscf.lib import logger
from pyscf.data import nist

class HF(pbc.scf.hf.RHF):
    '''
    NEO-HF for periodic systems at a single k-point

    Example:

    >>> from pyscf import neo
    >>> cl = neo.pbc.Cell()
    >>> cl.build(a = '5.21 0 0; 0 5.21 0; 0 0 5.21', atom = 'H 0 0 0; H 2.105 2.105 2.105',
                    basis = 'sto3g', quantum_nuc=[0,1])
    >>> mf = neo.pbc.HF(cl)
    >>> mf.scf()
    '''

    def __init__(self, cell):
        pbc.scf.hf.RHF.__init__(self, cell)
        self.verbose = 4
        self.cell = cell

        # set up the Hamiltonian for electrons
        self.mf_elec = pbc.scf.hf.RHF(self.cell.elec).mix_density_fit()
        #self.mf_elec.with_df = pbc.df.DF(self.cell.elec)
        self.dm_elec = self.mf_elec.get_init_guess(key='1e')
        self.mf_elec.get_hcore = self.get_hcore_elec

        # set up the Hamiltonian for quantum nuclei
        self.mf_nuc = [None] * self.cell.nuc_num
        self.dm_nuc = [None] * self.cell.nuc_num

        for i in range(len(self.cell.nuc)):
            self.mf_nuc[i] = pbc.scf.hf.RHF(self.cell.nuc[i]).mix_density_fit()
            #self.mf_nuc[i].with_df = pbc.df.DF(self.cell.nuc[i])
            self.mf_nuc[i].occ_state = 0 # for delta-SCF
            self.mf_nuc[i].get_occ = self.get_occ_nuc(self.mf_nuc[i])
            self.mf_nuc[i].get_hcore = self.get_hcore_nuc
            self.mf_nuc[i].get_veff = self.get_veff_nuc
            self.mf_nuc[i].get_init_guess = self.get_init_guess_nuc
            self.dm_nuc[i] = self.get_init_guess_nuc(self.mf_nuc[i])

    def get_hcore_nuc(self, cell, kpts=numpy.zeros(3)):
        'get the core Hamiltonian for quantum nucleus.'

        ia = cell.atom_index
        mass = self.cell.mass[ia] * nist.ATOMIC_MASS/nist.E_MASS # the mass of quantum nucleus in a.u.
        charge = self.cell.atom_charge(ia)

        # nuclear kinetic energy and Coulomb interactions with classical nuclei
        h = cell.pbc_intor('int1e_kin', hermi=1, kpts=kpts)/mass
        #h -= cell.pbc_intor('int1e_nuc', hermi=1, kpts=kpts)*charge
        h -= pbc.scf.hf.get_nuc(cell, kpts)*charge

        # Coulomb interactions between quantum nucleus and electrons (using density fitting)
        #mesh = [10, 10, 10]

        df_elec = pbc.df.MDF(self.cell.elec) # other DF methods?
        df_nuc = pbc.df.MDF(cell)
        df_nuc.mesh = df_elec.mesh = cell.mesh = [20, 20, 20]# make sure girds are enough

        # number of AO
        nao = self.cell.elec.nao_nr()
        nao_nuc = cell.nao_nr()

        #TODO: try ft_ao
        elec_ao_pair_G = df_elec.get_ao_pairs_G(kpts=numpy.zeros((2,3))).reshape(-1, nao, nao) # Note: kpoints
        elec_density_G = numpy.einsum('nij,ji->n', elec_ao_pair_G, self.dm_elec)
        nuc_ao_pair_G = df_nuc.get_ao_pairs_G(kpts=numpy.zeros((2,3))).reshape(-1, nao_nuc, nao_nuc)

        # set up G vectors
        coulG = tools.get_coulG(cell, k=kpts)

        # Coulomb interaction between quantum nucleus and electrons in reciprocal girds
        # check the formula
        # read pbc.df.df_jk
        # use AO loop?
        veff = - numpy.einsum('n,nij,n->ij', coulG, nuc_ao_pair_G, elec_density_G) * charge

        '''
        # Coulomb interactions between quantum nuclei
        for j in range(len(self.dm_nuc)):
            ja = self.mol.nuc[j].atom_index
            if ja != ia and isinstance(self.dm_nuc[j], numpy.ndarray):
                veff += scf.jk.get_jk((mole, mole, self.mol.nuc[j], self.mol.nuc[j]),
                                   self.dm_nuc[j], scripts='ijkl,lk->ij')*charge*self.mol.atom_charge(ja)
        '''
        return h + veff

    def get_occ_nuc(self, mf_nuc):
        def get_occ(nuc_energy, nuc_coeff):
            'label the occupation for quantum nucleus'

            e_idx = numpy.argsort(nuc_energy)
            nuc_occ = numpy.zeros(nuc_energy.size)
            nuc_occ[e_idx[mf_nuc.occ_state]] = 1

            return nuc_occ
        return get_occ

    def get_init_guess_nuc(self, mf_nuc, key=None):
        '''Generate initial guess density matrix for quantum nuclei

           Returns:
            Density matrix, 2D ndarray
        '''

        h1n = self.get_hcore_nuc(mf_nuc.cell)
        s1n = mf_nuc.cell.pbc_intor('int1e_ovlp', hermi=0)
        nuc_energy, nuc_coeff = scf.hf.eig(h1n, s1n)
        nuc_occ = mf_nuc.get_occ(nuc_energy, nuc_coeff)

        return scf.hf.make_rdm1(nuc_coeff, nuc_occ)

    def get_hcore_elec(self, cell=None):
        'get the core Hamiltonian for electrons (added e-n Coulomb)'
        if cell == None:
            cell = self.cell.elec
        j = 0
        for i in range(self.cell.nuc_num):
            ia = self.cell.nuc[i].atom_index
            charge = self.cell.atom_charge(ia)

            mesh = [20, 20, 20]

            df_elec = pbc.df.MDF(cell) # other DF methods?
            df_nuc = pbc.df.MDF(self.cell.nuc[i])
            df_nuc.mesh = df_elec.mesh = mesh # make sure girds are enough

            # number of AO
            nao = cell.nao_nr()
            nao_nuc = self.cell.nuc[i].nao_nr()

            elec_ao_pair_G = df_elec.get_ao_pairs_G(kpts=numpy.zeros((2,3))).reshape(-1, nao, nao) # Note: kpoints
            nuc_ao_pair_G = df_nuc.get_ao_pairs_G(kpts=numpy.zeros((2,3))).reshape(-1, nao_nuc, nao_nuc)
            nuc_density_G = numpy.einsum('nij,ji->n', nuc_ao_pair_G, self.dm_nuc[i])
            # set up G vectors
            coulG = tools.get_coulG(cell, mesh=mesh)

            # Coulomb interaction between quantum nucleus and electrons in reciprocal girds
            # check the formula
            # read pbc.df.df_jk
            j -= numpy.einsum('n,nij,n->ij', coulG, elec_ao_pair_G, nuc_density_G) * charge

        return pbc.scf.hf.get_hcore(cell) + j

    def get_veff_nuc(self, cell, dm, dm_last=None, vhf_last=None, hermi=1, vhfopt=None):
        'NOTE: Only for single quantum proton system.'
        return numpy.zeros((cell.nao_nr(), cell.nao_nr()))


if __name__ == '__main__':
    from pyscf import neo
    cl = neo.pbc.Cell()
    cl.build(a='5.21 0 0; 0 5.21 0; 0 0 5.21', atom='H 0 0 0; H 2.105 2.105 2.105', basis='ccpvdz', quantum_nuc = [0])
    mf = neo.pbc.HF(cl)
    mf.mf_nuc[0].verbose = 5
    mf.mf_nuc[0].scf()
