#!/usr/bin/env python

'''
Non-relativistic restricted Kohn-Sham for NEO-DFT
'''
import numpy
from pyscf import dft, lib
from pyscf.lib import logger
from pyscf.dft.numint import eval_ao, eval_rho, _scale_ao, _dot_ao_ao
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

    def __init__(self, mol, restrict=True):
        HF.__init__(self, mol, restrict)
        
        if restrict == True:
            self.mf_elec = dft.RKS(mol.elec)
        else:
            self.mf_elec = dft.UKS(mol.elec)

        self.mf_elec.get_hcore = self.get_hcore_elec
        self.mf_elec.get_veff = self.get_veff_elec
        self.mf_elec.xc = 'b3lyp' # use b3lyp as the default xc functional for electrons
        self.mf_elec.grids.level = 3 # high density grids are needed since nuclei is more localized
        self.dm_elec = self.mf_elec.get_init_guess(key='1e')

        # set up the Hamiltonian for each quantum nuclei
        for i in range(len(self.mol.nuc)):
            ia = self.mol.nuc[i].atom_index
            if self.mol.atom_symbol(ia) == 'H': # only support electron-proton correlation
                self.mf_nuc[i] = dft.RKS(self.mol.nuc[i])
                self.mf_nuc[i].get_init_guess = self.get_init_guess_nuc
                self.mf_nuc[i].get_hcore = self.get_hcore_nuc
                self.mf_nuc[i].nuc_state = 0
                self.mf_nuc[i].get_occ = self.get_occ_nuc(self.mf_nuc[i])
                #self.mf_nuc[i].grids.level = 9 # high density grids are needed since nuclei is more localized
                self.mf_nuc[i].get_veff = self.get_veff_nuc

            self.dm_nuc[i] = self.get_init_guess_nuc(self.mf_nuc[i])

    def eval_xc_nuc(self, rho_e, rho_n):
        'evaluate e_xc and v_xc of proton on a grid (epc17)'
        a = 2.35
        b = 2.4
        c = 3.2

        rho_product = numpy.multiply(rho_e, rho_n)
        denominator = a - b*numpy.sqrt(rho_product) + c*rho_product
        exc = - numpy.multiply(rho_product, 1/denominator)

        denominator = numpy.square(denominator)
        numerator = -a * rho_e + numpy.multiply(numpy.sqrt(rho_product), rho_e)*b/2
        vxc = numpy.multiply(numerator, 1/denominator)

        return exc, vxc

    def eval_xc_elec(self, rho_e, rho_n):
        'evaluate e_xc and v_xc of electrons on a grid (only the epc part)'
        a = 2.35
        b = 2.4
        c = 3.2

        rho_product = numpy.multiply(rho_e, rho_n)
        denominator = a - b*numpy.sqrt(rho_product) + c*rho_product
        exc = - numpy.multiply(rho_product, 1/denominator)

        denominator = numpy.square(denominator)
        numerator = -a * rho_n + numpy.multiply(numpy.sqrt(rho_product), rho_n)*b/2
        vxc = numpy.multiply(numerator, 1/denominator)

        return exc, vxc
            
    def get_veff_nuc(self, mol, dm, dm_last=None, vhf_last=None, hermi=1, vhfopt=None):
        'get the effective potential for proton of NEO-DFT'
        grids = self.mf_elec.grids
        coords = grids.coords 

        ao_elec = eval_ao(self.mol.elec, coords)
        rho_elec = eval_rho(self.mol.elec, ao_elec, self.dm_elec)

        ao_nuc = eval_ao(mol, coords)
        rho_nuc = eval_rho(mol, ao_nuc, dm)
        
        exc, vxc = self.eval_xc_nuc(rho_elec, rho_nuc)

        nnuc = 0 
        excsum = 0
        nao = mol.nao_nr()
        shls_slice = (0, mol.nbas)
        ao_loc = mol.ao_loc_nr()
        vmat = numpy.zeros((nao, nao))

        ni = self.mf_elec._numint

        for ao, mask, weight, coords in ni.block_loop(mol, grids, nao):
            den = rho_nuc * weight
            nnuc += den.sum()
            excsum += numpy.dot(den, exc)
            aow = _scale_ao(ao_nuc, .5*weight*vxc) # beta: *0.5 because vmat + vmat.T ?
            vmat += _dot_ao_ao(mol, ao_nuc, aow, mask, shls_slice, ao_loc)
        
        logger.debug(self, 'the number of nuclei: %.5f', nnuc)
        vmat = lib.tag_array(vmat, exc=excsum, ecoul=0, vj=0, vk=0)
        return vmat

    def get_veff_elec(self, mol, dm, dm_last=0, vhf_last=0, hermi=1):
        'get the effective potential for electrons of NEO-DFT'

        grids = self.mf_elec.grids
        coords = grids.coords 
        if coords is None:
            grids.build(with_non0tab = True)
            #if self.mf_elec.small_rho_cutoff > 1e-20:
            #    grids = dft.rks.prune_small_rho_grids_(self.mf_elec, mol, dm, grids)
            coords = grids.coords 

        ao_elec = eval_ao(mol, coords)
        rho_elec = eval_rho(mol, ao_elec, dm)

        N = coords.shape[0] # number of grids
        exc = vxc = numpy.zeros(N)
        
        for i in range(len(self.mol.nuc)):
            ia = self.mol.nuc[i].atom_index
            if self.mol.atom_symbol(ia) == 'H':
                ao_nuc = eval_ao(self.mol.nuc[i], coords)
                rho_nuc = eval_rho(self.mol.nuc[i], ao_nuc, self.dm_nuc[i])

                exc_i, vxc_i = self.eval_xc_elec(rho_elec, rho_nuc)

                exc += exc_i
                vxc += vxc_i

        nelec = 0
        excsum = 0
        nao = mol.nao_nr()
        shls_slice = (0, mol.nbas)
        ao_loc = mol.ao_loc_nr()
        vmat = numpy.zeros((nao, nao))

        ni = self.mf_elec._numint

        for ao, mask, weight, coords in ni.block_loop(mol, grids, nao):
            den = rho_elec * weight
            nelec += den.sum()
            excsum += numpy.dot(den, exc)
            aow = _scale_ao(ao_elec, .5*weight*vxc) # beta: *0.5 because vmat + vmat.T ?
            vmat += _dot_ao_ao(mol, ao_elec, aow, mask, shls_slice, ao_loc)

        #logger.debug(self, 'The number of electrons by numerical intergration: %.5f', nelec)
        vxc = dft.rks.get_veff(self.mf_elec, mol, dm, dm_last, vhf_last, hermi)
        ecoul = vxc.ecoul
        exc = vxc.exc
        vj = vxc.vj
        vk = vxc.vk
        vxc = lib.tag_array(vxc + vmat, ecoul=ecoul, exc = exc, vj = vj, vk = vk)
        return vxc

    def energy_tot(self):
        'Total energy of NEO-DFT'
        E_tot = 0

        self.dm_elec = self.mf_elec.make_rdm1()
        for i in range(len(self.mf_nuc)):
            self.dm_nuc[i] = self.mf_nuc[i].make_rdm1()

        h1e = self.mf_elec.get_hcore(self.mf_elec.mol)
        if self.restrict == False:
            e1 = numpy.einsum('ij,ji', h1e, self.dm_elec[0] + self.dm_elec[1])
        else:
            e1 = numpy.einsum('ij,ji', h1e, self.dm_elec)
        logger.debug(self, 'Energy of e1: %s', e1)

        vhf = self.mf_elec.get_veff(self.mf_elec.mol, self.dm_elec)
        if self.restrict == False:
            e_coul = (numpy.einsum('ij,ji', vhf[0], self.dm_elec[0]) +
                    numpy.einsum('ij,ji', vhf[1], self.dm_elec[1])) * .5 
        else:
            e_coul = numpy.einsum('ij,ji', vhf, self.dm_elec)
        logger.debug(self, 'Energy of e-e Coulomb interactions: %s', e_coul)

        E_tot += self.mf_elec.energy_elec(dm = self.dm_elec, h1e = h1e, vhf = vhf)[0] 

        for i in range(len(self.mf_nuc)):
            ia = self.mf_nuc[i].mol.atom_index
            h1n = self.mf_nuc[i].get_hcore(self.mf_nuc[i].mol)
            n1 = numpy.einsum('ij,ji', h1n, self.dm_nuc[i])
            logger.debug(self, 'Energy of %s: %s', self.mol.atom_symbol(ia), n1)
            E_tot += n1
            if self.mol.atom_symbol(ia) == 'H':
                veff = self.mf_nuc[i].get_veff(self.mf_nuc[i].mol, self.dm_nuc[i])
                E_tot += veff.exc

        E_tot =  E_tot - self.elec_nuc_coulomb(self.dm_elec, self.dm_nuc) - self.nuc_nuc_coulomb(self.dm_nuc) + self.mf_elec.energy_nuc() # substract repeatedly counted terms

        return E_tot




