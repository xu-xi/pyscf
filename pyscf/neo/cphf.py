#!/usr/bin/env python

'''
Coupled perturbed Hartree-Fock for constrained nuclear-electronic orbital method
'''
import numpy
from pyscf import lib, gto, scf
from functools import reduce

class CPHF(lib.StreamObject):
    def __init__(self, scf_method):
        self.mol = scf_method.mol
        self.base = scf_method
        self.atmlst = range(self.mol.natm)

        #nset = 3*len(self.atmlst)

    def ao2mo(self, mf, mat):
        return numpy.asarray([reduce(numpy.dot, (mf.mo_coeff.T, x, mf.mo_coeff[:,mf.mo_occ>0])) for x in mat])

    def mo12ne(self, mo1):
        'transfer mo1 from a 1-d array to mo1_e, mo1_n and f1 with proper shape'

        mo_coeff = self.base.mf_elec.mo_coeff
        mo_occ = self.base.mf_elec.mo_occ
        nao, nmo = mo_coeff.shape
        mocc = mo_coeff[:,mo_occ>0]
        nocc = mocc.shape[1]
        nset = 3*len(self.atmlst)

        n_e = nset * nmo * nocc
        index = n_e
        mo1 = mo1.ravel()
        mo1_e = mo1[:index].reshape(nset, nmo, nocc)
        
        mo1_n = []
        for i in range(len(self.mol.nuc)):
            mo_coeff = self.base.mf_nuc[i].mo_coeff
            mo_occ = self.base.mf_nuc[i].mo_occ
            mocc = mo_coeff[:,mo_occ>0]
            nocc = mocc.shape[1]
            nao, nmo = mo_coeff.shape
            add = nset * nmo * nocc # nocc = 1 actually
            mo1_n.append(mo1[index: index + add].reshape(nset, nmo, nocc))
            index += add

        f1 = []
        for i in range(len(self.mol.nuc)):
            add = nset*3
            f1.append(mo1[index:index + add].reshape(nset, 3))
            index += add

        return mo1_e, mo1_n, f1

   
    def get_A_e(self, mo1):
        'get the response matrix for electrons'

        mf_e = self.base.mf_elec
        mo_coeff_e = mf_e.mo_coeff
        mo_occ_e = mf_e.mo_occ
        nao_e, nmo_e = mo_coeff_e.shape
        nset = 3*len(self.atmlst)

        e_a = mf_e.mo_energy[mo_occ_e == 0]
        e_i = mf_e.mo_energy[mo_occ_e > 0]
        e_ai = 1 / lib.direct_sum('a-i->ai', e_a, e_i)

        mo1_e, mo1_n, f1 = self.mo12ne(mo1)

        # calculate A^e * U^e
        vresp = mf_e.gen_response(mo_coeff_e, mo_occ_e, hermi=1)
        dm1 = numpy.empty((nset, nao_e, nao_e))
        for i, x in enumerate(mo1_e):
            dm = reduce(numpy.dot, (mo_coeff_e.T, x*2, mo_coeff_e[:, mo_occ_e>0].T)) # *2 for double occupancy
            dm1[i] = dm + dm.T
        v1 = vresp(dm1)
        v1vo = numpy.empty_like(mo1_e)
        for i, x in enumerate(v1):
            v1vo[i] = reduce(numpy.dot, (mo_coeff_e.T, x, mo_coeff_e[:, mo_occ_e>0]))

        v1vo[:,mo_occ_e==0,:] *= -e_ai
        v1vo[:,mo_occ_e>0,:] = 0

        # calculate C^e * U^n
        for i in range(len(self.mol.nuc)):
            mo_coeff_n = self.base.mf_nuc[i].mo_coeff
            mo_occ_n = self.base.mf_nuc[i].mo_occ
            nao_n, nmo_n = mo_coeff_n.shape

            j = self.mol.nuc[i].atom_index
            v1en_ao = -2 * gto.conc_mol(self.mol.elec, self.mol.nuc[i]).intor('int2e_sph').reshape([nao_e + nao_n]*4) * self.mol._atm[j,0] # sign
            v1en_ao = numpy.einsum('mnpq, mi, na ->pqia', v1en_ao[:nao_e, :nao_e, nao_e:, nao_e:], mo_coeff_e, mo_coeff_e[:, mo_occ_e>0])
            C = numpy.einsum('pk,pqia,ql-> klia', mo_coeff_n, v1en_ao, mo_coeff_n[:,mo_occ_n>0])
            C = numpy.einsum('klia, ckl->cia', C, mo1_n[i]) # test: cmn or cnm ?
            C[:,mo_occ_e==0,:] *= -e_ai
            C[:,mo_occ_e>0,:] = 0
            v1vo += C

        #print('v1vo.shape', v1vo.shape)
        return v1vo

    def get_A_n(self, i, mo1):
        'get the response of quantum nuclei'
        mf_n = self.base.mf_nuc[i]
        mo_coeff_n = mf_n.mo_coeff
        mo_occ_n = mf_n.mo_occ
        nao_n, nmo_n = mo_coeff_n.shape

        e_a = mf_n.mo_energy[mo_occ_n == 0]
        e_i = mf_n.mo_energy[mo_occ_n > 0]
        e_ai = 1 / lib.direct_sum('a-i->ai', e_a, e_i)

        mo1_e, mo1_n, f1 = self.mo12ne(mo1)

        # calculate C^e^T * U^e
        mf_e = self.base.mf_elec
        mo_coeff_e = mf_e.mo_coeff
        mo_occ_e = mf_e.mo_occ
        nao_e, nmo_e = mo_coeff_e.shape

        j = self.mol.nuc[i].atom_index
        v1ne_ao = -2 * gto.conc_mol(self.mol.elec, self.mol.nuc[i]).intor('int2e_sph').reshape([nao_e + nao_n]*4) * self.mol._atm[j,0]
        v1ne_ao = numpy.einsum('pqmn, mi, na->iapq', v1ne_ao[:nao_e, :nao_e, nao_e:, nao_e:], mo_coeff_n, mo_coeff_n[:,mo_occ_n>0]) 
        v1ne = numpy.einsum('iapq, pk, ql->iakl', v1ne_ao, mo_coeff_e, mo_coeff_e[:,mo_occ_e>0])
        C = numpy.einsum('iakl, ckl-> cia', v1ne, mo1_e)

        for j in range(self.mol.nuc_num):
            if j != i:
                mf_j = self.base.mf_nuc[j]
                mo_coeff_j = mf_j.mo_coeff
                mo_occ_j = mf_j.mo_occ
                nao_j, nmo_j = mo_coeff_j.shape
                v_nn_ao = gto.conc_mol(self.mol.nuc[j], self.mol.nuc[i]).intor('int2e_sph').reshape([nao_j + nao_n]*4) * self.mol._atm[j,0] * self.mol._atm[i,0] # sign
                v_nn_ao = numpy.einsum('mnpq, pi, qa->mnia', v_nn_ao[:nao_j, :nao_j, nao_j:, nao_j:], mo_coeff_n, mo_coeff_n[:,mo_occ_n>0])
                #v_nn = scf.jk.get_jk((self.mol.nuc[j], self.mol.nuc[j], self.mol.nuc[i], self.mol.nuc[i]), dm_nuc, scripts='mnpq, ipaq->mnia', aosym='s4') # test: symmetry
                v_nn = numpy.einsum('mnia, mk, nl->klia', v_nn_ao, mo_coeff_j, mo_coeff_j[:,mo_occ_j>0])
                C += numpy.einsum('klia, ckl->cia', v_nn, mo1_n[j])

        # calculate R * F
        R_ao = self.mol.nuc[i].intor_symmetric('int1e_r', comp=3)
        R = numpy.einsum('xmn, mi, na->xia', R_ao, mo_coeff_n, mo_coeff_n[:, mo_occ_n>0])
        C += numpy.einsum('xia, cx->cia', R, f1[i])

        #print('get_A_n', C.shape)
        return C


    def get_R(self, i, mo1):
        mf_n = self.base.mf_nuc[i]
        mo_coeff = mf_n.mo_coeff
        mo_occ = mf_n.mo_occ
        dm_nuc = numpy.einsum('mi, na->mina', mo_coeff, mo_coeff[:, mo_occ>0])

        mo1_e, mo1_n, f1 = self.mo12ne(mo1)

        R_ao = self.mol.nuc[i].intor_symmetric('int1e_r', comp=3)
        R = numpy.einsum('xmn, mina->xia', R_ao, dm_nuc)
        R = numpy.einsum('xia, cia ->cx', R, mo1_n[i])

        # minus identity matrix
        R -= f1[i]
        return R


    def full_response(self, mo1):
        'set up the full matrix and multiply by mo1'
        response = self.get_A_e(mo1).ravel()
        for i in range(self.mol.nuc_num):
            response = numpy.concatenate((response, self.get_A_n(i, mo1).ravel()))
        for i in range(self.mol.nuc_num):
            response = numpy.concatenate((response, self.get_R(i, mo1).ravel()))
        return response

    def get_Bmat_elec(self, mf_elec, a):
        'get the B matrix for electrons w.r.t. the a-th nuclei'
        mol = mf_elec.mol

        occidx = mf_elec.mo_occ > 0
        viridx = mf_elec.mo_occ == 0
        e_a = mf_elec.mo_energy[viridx]
        e_i = mf_elec.mo_energy[occidx]
        e_ai = 1 / lib.direct_sum('a-i->ai', e_a, e_i)
        nvir, nocc = e_ai.shape
        nao, nmo = mf_elec.mo_coeff.shape

        shl0, shl1, p0, p1 = self.mol.aoslice_by_atom()[a, :]

        h1ao = numpy.zeros((3,nao,nao))
        s1ao = numpy.zeros((3,nao,nao))
        v1en_ao = numpy.zeros((3,nao,nao))

        hobj = mf_elec.Hessian()
        h1ao += hobj.make_h1(mf_elec.mo_coeff, mf_elec.mo_occ)[a]

        s1a = -mol.intor('int1e_ipovlp', comp=3)
        s1ao[:,p0:p1] += s1a[:,p0:p1]
        s1ao[:,:,p0:p1] += s1a[:,p0:p1].transpose(0,2,1)

        if self.mol.quantum_nuc[a] == True:
            for i in range(len(self.mol.nuc)):
                if self.mol.nuc[i].atom_index == a:
                    v1en_ao += scf.jk.get_jk((self.mol.nuc[i], self.mol.nuc[i], self.mol.elec, self.mol.elec), self.base.dm_nuc[i], scripts='ijkl,ji->kl', intor='int2e_ip1_sph', comp=3)*self.mol._atm[a,0]
                    v1en_ao += v1en_ao.transpose(0, 2, 1)
                    #shls_slice = (0, mol.nbas) + (shl0, shl1)  + (0, self.mol.nuc[i].nbas)*2
                    v1en_ao2 = scf.jk.get_jk((self.mol.elec, self.mol.elec, self.mol.nuc[i], self.mol.nuc[i]), self.base.dm_nuc[i], scripts='ijkl,lk->ij', intor='int2e_ip1_sph', comp=3)*self.mol._atm[a,0] #test
                    v1en_ao[:,p0:p1] += v1en_ao2[:,p0:p1]
                    v1en_ao[:,:,p0:p1] += v1en_ao2[:,p0:p1].transpose(0,2,1)

        B = self.ao2mo(mf_elec, h1ao - v1en_ao)  - self.ao2mo(mf_elec, s1ao)*e_i
        B[:,viridx] *= -e_ai
        B[:,occidx] = - self.ao2mo(mf_elec, s1ao)[:, occidx] * .5
        #print(B.shape)
        return B

    def get_Bmat_nuc(self, mf_nuc, a):
        'get the B matrix for quantum nuclei w.r.t the a-th nuclei'
        mol = mf_nuc.mol
        i = mol.atom_index

        occidx = mf_nuc.mo_occ > 0
        viridx = mf_nuc.mo_occ == 0
        e_i = mf_nuc.mo_energy[occidx]
        e_a = mf_nuc.mo_energy[viridx]
        e_ai = 1 / lib.direct_sum('a-i->ai', e_a, e_i)
        nvir, nocc = e_ai.shape
        nao, nmo = mf_nuc.mo_coeff.shape

        h1ao = numpy.zeros((3,nao,nao))
        v1ne_ao = numpy.zeros((3,nao,nao))
        v1nn_ao = numpy.zeros((3,nao,nao))
        s1ao = numpy.zeros((3,nao,nao))
        f1ao = numpy.zeros((3,nao,nao))

        if i == a:
            mass = 1836.15267343 * self.mol.mass[i]
            h1ao += mol.intor('int1e_ipkin', comp=3)/mass
            h1ao -= mol.intor('int1e_ipnuc', comp=3)*self.mol._atm[i,0]
            h1ao += h1ao.transpose(0, 2, 1)
    
            v1ne_ao += scf.jk.get_jk((mol, mol, self.mol.elec, self.mol.elec), self.base.dm_elec, scripts='ijkl,lk->ij', intor='int2e_ip1_sph', comp=3)*self.mol._atm[i,0]
            v1ne_ao += v1ne_ao.transpose(0, 2, 1)
            
            for j in range(len(self.mol.nuc)):
                k = self.mol.nuc[j].atom_index
                if k != i:
                    v1nn_ao += scf.jk.get_jk((mol, mol, self.mol.nuc[j], self.mol.nuc[j]), self.base.dm_nuc[j], scripts='ijkl,lk->ij', intor='int2e_ip1_sph', comp=3)*self.mol._atm[i,0]*self.mol._atm[k,0]
            v1nn_ao += v1nn_ao.transpose(0, 2, 1)

            f1ao += numpy.einsum('ijkl,j->ikl', -mol.intor('int1e_irp').reshape(3, 3, nao, nao), self.base.f[i]) #beta
            f1ao += f1ao.transpose(0, 2, 1)

            s1ao -= mol.intor('int1e_ipovlp', comp=3) 
            s1ao += s1ao.transpose(0, 2, 1)

        else:
            if self.mol.quantum_nuc[a] == True:
                for j in range(len(self.mol.nuc)):
                    k = self.mol.nuc[j].atom_index
                    if k == a:
                        v1nn_ao += scf.jk.get_jk((self.mol.nuc[j], self.mol.nuc[j], mol, mol), self.base.dm_nuc[j], scripts='ijkl,ji->kl', intor='int2e_ip1_sph', comp=3)*self.mol._atm[i,0]*self.mol._atm[k,0]
                v1nn_ao += v1nn_ao.transpose(0, 2, 1)
            else:
                with mol.with_rinv_as_nucleus(a):
                    vrinv = mol.intor('int1e_iprinv', comp=3) # <\nabla|1/r|>
                    vrinv *= (mol.atom_charge(a)*self.mol._atm[i,0])
                h1ao = vrinv + vrinv.transpose(0, 2, 1)

        shl0, shl1, p0, p1 = self.mol.aoslice_by_atom()[a, :]
        shls_slice = (0, self.mol.elec.nbas) + (shl0, shl1) + (0, mol.nbas)*2
        v1ne_ao += scf.jk.get_jk((self.mol.elec, self.mol.elec, mol, mol), self.base.dm_elec[p0:p1], scripts='ijkl,ji->kl', intor='int2e_ip1_sph', shls_slice=shls_slice, comp=3)*self.mol._atm[i,0] #test
        v1ne_ao += v1ne_ao.transpose(0, 2, 1)

        B = self.ao2mo(mf_nuc, h1ao - v1ne_ao + v1nn_ao)  - self.ao2mo(mf_nuc, s1ao)*e_i
        B[:,viridx] *= -e_ai
        B[:,occidx] = -self.ao2mo(mf_nuc, s1ao)[:,occidx] * .5
        return B

    def kernel(self, max_cycle=20, tol=1e-9, hermi=False):
        '''
        CPHF solver for cNEO.

        mo1_e: 1st order response of electronic orbital coefficients to the displacements of nuclei.
        e1: 1st order response of electronic eigenvalues to the displacements of nuclei.
        mo1_n: 1st order response of nuclear orbital coefficients to the displacements of nuclei.
        n1: 1st order response of nuclear eigenvalues to the displacements of nuclei.
        '''
        #vind_vo = self.get_Amat()
        
        mo1base = numpy.array([])
        for j in range(self.mol.natm):
            mo1base = numpy.concatenate((mo1base, self.get_Bmat_elec(self.base.mf_elec, j).ravel()))
        for i in range(len(self.mol.nuc)):
            for j in range(self.mol.natm):
                mo1base = numpy.concatenate((mo1base, self.get_Bmat_nuc(self.base.mf_nuc[i],j).ravel()))
        for i in range(len(self.mol.nuc)):
            for j in range(self.mol.natm):
                mo1base = numpy.concatenate((mo1base, numpy.identity(3).ravel()))
        print('mo1base', len(mo1base))

        mo1 = lib.krylov(self.full_response, mo1base, tol=tol, max_cycle=max_cycle, hermi=hermi)
        print(mo1.shape)
        #return mo1_e, e1, mo1_n, n1 


