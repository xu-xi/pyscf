#!/usr/bin/env python

'''
Analytical Hessian for constrained nuclear-electronic orbitals
'''
import numpy
from pyscf import lib
from pyscf import scf
from pyscf.neo.cphf import CPHF
from functools import reduce

class Hessian(lib.StreamObject):
    '''
    Example:

    >>> mol = neo.Mole()
    >>> mol.build(atom = 'H 0 0 0.00; C 0 0 1.064; N 0 0 2.220', basis = 'ccpvdz')
    >>> mf = neo.CDFT(mol)
    >>> mf.mf_elec.xc = 'b3lyp'
    >>> mf.scf()

    >>> g = neo.Hessian(mf)
    >>> g.kernel()
    '''

    def __init__(self, scf_method):
        self.mol = scf_method.mol
        self.base = scf_method
        self.atmlst = range(self.mol.natm)
        self.de = numpy.zeros((0, 0, 3, 3))

    def hess_elec(self, mo1_e, e1_e):
        'Hessian of electrons and classic nuclei in cNEO'
        hobj = self.base.mf_elec.Hessian()
        return hobj.hess_elec(mo1 = mo1_e, mo_e1 = e1_e) + hobj.hess_nuc()
 
    def hess_elec_nuc1(self, i, mo1_n, mo1_e):
        'part of hessian for Coulomb interactions between electrons and the i-th quantum nuclei'
        de2 = numpy.zeros((self.mol.natm, self.mol.natm, 3, 3))
        aoslices = self.mol.aoslice_by_atom()

        dm0_e = self.base.dm_elec
        mo_coeff_e = self.base.mf_elec.mo_coeff
        mo_occ_e = self.base.mf_elec.mo_occ
        mocc_e = mo_coeff_e[:, mo_occ_e >0]

        mf_nuc = self.base.mf_nuc[i]
        nuc = mf_nuc.mol
        index = nuc.atom_index
        charge = self.mol._atm[index, 0]
        mo_coeff_n = mf_nuc.mo_coeff
        mo_occ_n = mf_nuc.mo_occ
        mocc_n = mo_coeff_n[:, mo_occ_n >0]
        dm0_n = self.base.dm_nuc[i]

        v1en_ao = -scf.jk.get_jk((self.mol.elec, self.mol.elec, nuc, nuc), dm0_n, scripts='ijkl,lk->ij', intor='int2e_ip1_sph', comp=3)
        v1en_ao2 = -scf.jk.get_jk((self.mol.elec, self.mol.elec, nuc, nuc), dm0_e, scripts='ijkl,ji->kl', intor='int2e_ip1_sph', comp=3)
        for i0, ia in enumerate(self.atmlst):
            shl0, shl1, p0, p1 = aoslices[ia]
            if index == ia:
                v1en_ao -= scf.jk.get_jk((nuc, nuc, self.mol.elec, self.mol.elec), dm0_n, scripts='ijkl,ji->kl', intor='int2e_ip1_sph', comp=3)
                v1en_ao2 -= scf.jk.get_jk((nuc, nuc, self.mol.elec, self.mol.elec), dm0_e, scripts='ijkl,lk->ij', intor='int2e_ip1_sph', comp=3)

            v1en_ao += v1en_ao.transpose(0, 2, 1)
            v1en_ao2 += v1en_ao2.transpose(0, 2, 1)

            for j0 in range(i0+1):
                ja = self.atmlst[j0]
                dm1_e = numpy.einsum('ypi,qi->ypq', mo1_e[ja], mocc_e)
                de2[i0, j0] -= numpy.einsum('xpq, ypq->xy', v1en_ao[:,p0:p1], dm1_e[:,p0:p1])*charge 
                dm1_n = numpy.einsum('ypi, qi->ypq', mo1_n[ja], mocc_n)
                de2[i0, j0] -= numpy.einsum('xpq, ypq->xy', v1en_ao2, dm1_n)*charge

            for j0 in range(i0):
                de2[j0, i0] = de2[i0,j0].T
        return de2 

    def hess_elec_nuc2(self, i):
        'part of hessian for Coulomb interactions between electrons and the i-th quantum nuclei'
        mo_coeff_e = self.base.mf_elec.mo_coeff
        mo_occ_e = self.base.mf_elec.mo_occ
        dm0_e = self.base.dm_elec
        nao_e = dm0_e.shape[0]

        de2 = numpy.zeros((self.mol.natm, self.mol.natm, 3, 3))
        aoslices = self.mol.aoslice_by_atom()

        mf_nuc = self.base.mf_nuc[i]
        nuc = mf_nuc.mol
        index = nuc.atom_index
        charge = self.mol._atm[index, 0]
        mo_coeff_n = mf_nuc.mo_coeff
        mo_occ_n = mf_nuc.mo_occ
        dm0_n = self.base.dm_nuc[i]
        nao_n, nao_m = mf_nuc.mo_coeff.shape

        i = self.atmlst.index(index)
        hess_naa = scf.jk.get_jk((nuc, nuc, self.mol.elec, self.mol.elec), dm0_n, scripts='ijkl,ji->kl', intor='int2e_ipip1', comp=9)
        de2[i, i] -= numpy.einsum('xpq, pq->x', hess_naa, dm0_e).reshape(3, 3)*2

        hess_eaa = scf.jk.get_jk((self.mol.elec, self.mol.elec, nuc, nuc), dm0_n, scripts='ijkl,lk->ij', intor='int2e_ipip1', comp=9)

        for i0, ia in enumerate(self.atmlst):
            shl0, shl1, p0, p1 = aoslices[ia]
            shls_slice = (shl0, shl1) + (0, self.mol.elec.nbas) + (0, nuc.nbas)*2

            de2[i0, i0] -= numpy.einsum('xpq,pq->x', hess_eaa[:,p0:p1], dm0_e[p0:p1]).reshape(3,3)*2

            hess_naeb = scf.jk.get_jk((self.mol.elec, self.mol.elec, nuc, nuc), dm0_n, scripts='ijkl,lk->ij', intor='int2e_ip1ip2', comp=9) 
            de2[i, i0] -= numpy.einsum('xpq,pq->x', hess_naeb[:,p0:p1], dm0_e[p0:p1]).reshape(3,3)*4

            hess_eaeb = scf.jk.get_jk((self.mol.elec, self.mol.elec, nuc, nuc), dm0_n, scripts='ijkl,lk->ij', intor='int2e_ipvip1', shls_slice=shls_slice, comp=9)

            for j0, ja in enumerate(self.atmlst[:i0+1]):
                q0, q1 = aoslices[ja][2:]
                de2[i0,j0] -= numpy.einsum('xpq,pq->x', hess_eaeb[:,:,q0:q1], dm0_e[p0:p1,q0:q1]).reshape(3, 3)*2
                if index == ja:
                    hess_eanb = scf.jk.get_jk((self.mol.elec, self.mol.elec, nuc, nuc), dm0_n, scripts='ijkl,lk->ij', intor='int2e_ip1ip2', shls_slice=shls_slice, comp=9)
                    de2[i0,j0] -= numpy.einsum('xpq,pq->x', hess_eanb, dm0_e[p0:p1]).reshape(3, 3)*4

            for j0 in range(i0):
                de2[j0,i0] = de2[i0,j0].T
        return de2

    def hess_nuc1(self, i, mo1_n, e1_n, f1):
        'part of hessian for the i-th quantum nuclei'
        de2 = numpy.zeros((self.mol.natm, self.mol.natm, 3, 3))
        mf_nuc = self.base.mf_nuc[i]
        dm0_n = self.base.dm_nuc[i]
        index = mf_nuc.mol.atom_index
        mo_energy = mf_nuc.mo_energy
        mo_coeff = mf_nuc.mo_coeff
        mo_occ = mf_nuc.mo_occ
        mocc = mo_coeff[:, mo_occ>0]
        nao, nmo = mo_coeff.shape

        i0 = self.atmlst.index(index)

        h1a = -mf_nuc.mol.intor('int1e_ipkin', comp=3)/(1836.15267343 * self.mol.mass[index])
        h1a += mf_nuc.mol.intor('int1e_ipnuc', comp=3)*self.mol._atm[index,0]
        with mf_nuc.mol.with_rinv_as_nucleus(i0):
            vrinv = mf_nuc.mol.intor('int1e_iprinv', comp=3)
            vrinv *= (self.mol.atom_charge(i0)*self.mol._atm[index,0])
        h1a += vrinv
        h1a += h1a.transpose(0,2,1)

        r1a = -mf_nuc.mol.intor('int1e_irp')
        r1a += r1a.transpose(0,2,1)
        r1a = r1a.reshape(3, 3, nao, nao)

        s1a = -mf_nuc.mol.intor('int1e_ipovlp', comp=3)
        s1ao = s1a + s1a.transpose(0,2,1)
        s1oo = numpy.einsum('xpq,pi,qj->xij', s1ao, mocc, mocc)

        for j0 in range(i0+1):
            ja = self.atmlst[j0]
            de2[i0, j0] -= f1[ja]
           
            dm1_n = numpy.einsum('ypi,qi->ypq', mo1_n[ja], mocc)
            dm1_n += dm1_n.transpose(0,2,1) # test: or *2 for c.c.?

            de2[i0, j0] += numpy.einsum('xpq,ypq->xy', h1a, dm1_n)

            r1a_f = numpy.einsum('xypq, y->xpq', r1a, self.base.f[index])
            de2[i0, j0] += numpy.einsum('xpq,ypq->xy', r1a_f, dm1_n)

            r1a_dm = numpy.einsum('xzpq, pq->xz', r1a, dm0_n)
            de2[i0, j0] += numpy.einsum('xz, zy->xy', r1a_dm, f1[ja]) # test: yz or zy?

            dm1_n = numpy.einsum('ypi,qi,i->ypq', mo1_n[ja], mocc, mo_energy[mo_occ>0])
            dm1_n += dm1_n.transpose(0,2,1)
            de2[i0, j0] -= numpy.einsum('xpq,ypq->xy', s1ao, dm1_n)
            de2[i0, j0] -= numpy.einsum('xpq,ypq->xy', s1oo, e1_n[ja])
        for j0 in range(i0):
            de2[j0, i0] = de2[i0, j0].T

        return de2


    def hess_nuc2(self, i):
        'part of hessian for the i-th quantum nuclei'
        de2 = numpy.zeros((self.mol.natm, self.mol.natm, 3, 3))
        mf_nuc = self.base.mf_nuc[i]
        dm0_n = self.base.dm_nuc[i]
        index = mf_nuc.mol.atom_index
        mass = 1836.15267343 * self.mol.mass[index]
        charge = self.mol._atm[index, 0]
                
        mo_energy = mf_nuc.mo_energy
        mo_coeff = mf_nuc.mo_coeff
        mo_occ = mf_nuc.mo_occ
        mocc = mo_coeff[:, mo_occ>0]
        nao, nmo = mo_coeff.shape
        dme0 = numpy.einsum('pi,qi,i->pq', mocc, mocc, mo_energy[mo_occ>0])

        h1aa = mf_nuc.mol.intor('int1e_ipipkin', comp=9)/mass
        h1aa += mf_nuc.mol.intor('int1e_ipkinip', comp=9)/mass
        h1aa -= mf_nuc.mol.intor('int1e_ipipnuc', comp=9) * charge
        h1aa -= mf_nuc.mol.intor('int1e_ipnucip', comp=9) * charge
        h1aa = h1aa.reshape(3,3,nao,nao)
        h1aa += h1aa.transpose(0, 1, 3, 2)

        s1aa = mf_nuc.mol.intor('int1e_ipipovlp', comp=9).reshape(3,3,nao,nao)
        s1aa += mf_nuc.mol.intor('int1e_ipovlpip', comp=9).reshape(3,3,nao,nao)
        s1aa += s1aa.transpose(0, 1, 3, 2)

        r1aa = mf_nuc.mol.intor('int1e_ppr', comp=27).reshape(3, 3, 3, nao, nao)
        r1aa = numpy.einsum('xyzpq, z->xypq', r1aa, self.base.f[index])
        r1aa2 = mf_nuc.mol.intor('int1e_prp', comp=27).reshape(3, 3, 3, nao, nao)
        r1aa += numpy.einsum('xyzpq, y->xzpq', r1aa2, self.base.f[index])

        i0 = self.atmlst.index(index)

        # test: hessian for quantum-classsical nuclei replusions
        for j0 in range(i0+1):
            ja = self.atmlst[j0]
            if self.mol.quantum_nuc[ja] == False:
                with mf.nuc.mol.with_rinv_at_nucleus(ja):
                    rinvaa = mf.nuc.mol.intor('int1e_ipiprinv', comp=9).reshape(3, 3, nao, nao)
                    rinvaa += mf.nuc.mol.intor('int1e_iprinvip', comp=9).reshape(3, 3, nao, nao)
                    rinvaa *= (charge*self.mol._atm[ja, 0])
                    rinvaa += rinvaa.transpose(0, 1, 3, 2)
                h1aa += rinvaa

        de2[i0, i0] += numpy.einsum('xypq, pq->xy', h1aa, dm0_n)
        de2[i0, i0] -= numpy.einsum('xypq, pq->xy', s1aa, dme0)
        de2[i0, i0] += numpy.einsum('xypq, pq->xy', r1aa, dm0_n)
        return de2

    def hess_nuc_nuc1(self, mo1_n):
        'part of hessian for Coulomb interactions between quantum nuclei'
        de2 = numpy.zeros((self.mol.natm, self.mol.natm, 3, 3))
        for i in range(len(self.mol.nuc)):
            ia = self.mol.nuc[i].atom_index
            i0 = self.atmlst.index(index)
            for j in range(len(self.mol.nuc)):
                if j !=i:
                    ja = self.mol.nuc[j].atom_index
                    j0 = self.atmlst.index(index) 
                    mo_coeff = self.base.mf_nuc[j].mo_coeff
                    mo_occ = self.base.mf_nuc[j].mo_occ
                    mocc = mo_coeff[:, mo_occ>0]
                    dm1 = numpy.einsum('ypi,qi->ypq', mo1_n[ja], mocc)
                    v1 = scf.jk.get_jk((self.mol.nuc[i], self.mol.nuc[i], self.mol.nuc[j], self.mol.nuc[j]), self.scf.dm_nuc[i], scripts='ijkl,ji->kl', intor='int2e_ip1', comp=3)*self.mol._atm[ia,0]*self.mol._atm[ja,0]
                    v1 += scf.jk.get_jk((self.mol.nuc[j], self.mol.nuc[j], self.mol.nuc[i], self.mol.nuc[i]), self.scf.dm_nuc[i], scripts='ijkl,lk->ij', intor='int2e_ip1', comp=3)*self.mol._atm[ia,0]*self.mol._atm[ja,0]
                    de2[i0 ,j0] += numpy.einsum('xpq,ypq->xy', v1, dm1)*2

        return de2

    def hess_nuc_nuc2(self):
        'part of hessian for Coulomb interactions between quantum nuclei'
        de2 = numpy.zeros((self.mol.natm, self.mol.natm, 3, 3))
        for i in range(len(self.mol.nuc)):
            ia = self.mol.nuc[i].atom_index
            i0 = self.atmlst.index(index)
            v2_aa = 0
            for j in range(len(self.mol.nuc)):
                if j != i:
                    ja = self.mol.nuc[j].atom_index
                    j0 = self.atmlst.index(index)
                    v2_aa += scf.jk.get_jk((self.mol.nuc[i], self.mol.nuc[i], self.mol.nuc[j], self.mol.nuc[j]), self.scf.dm_nuc[j], scripts='ijkl,lk->ij', intor='int2e_ipip1', comp=9)*self.mol._atm[ia,0]*self.mol._atm[ja,0]
                    v2_ab = scf.jk.get_jk((self.mol.nuc[i], self.mol.nuc[i], self.mol.nuc[j], self.mol.nuc[j]), self.scf.dm_nuc[j], scripts='ijkl,lk->ij', intor='int2e_ip1ip2', comp=9)*self.mol._atm[ia,0]*self.mol._atm[ja,0]
                    de2[i0,j0] += numpy.einsum('xpq,pq->x', v2_ab, self.scf.dm_nuc[i]).reshape(3,3)*2
            de2[i0, i0] += numpy.einsum('xpq, pq->x', v1_aa, self.scf.dm_nuc[i]).reshape(3,3)*2

        return de2/2

    def kernel(self):
        cphf = CPHF(self.base)
        mo1_e, e1_e, mo1_n, e1_n, f1 = cphf.kernel()
        
        mo_coeff_e = self.base.mf_elec.mo_coeff
        mo_occ_e = self.base.mf_elec.mo_occ
        nao_e, nmo_e = mo_coeff_e.shape
        mocc_e = mo_coeff_e[:,mo_occ_e>0]
        nocc_e = mocc_e.shape[1]

        mo1_e = mo1_e.reshape(-1, 3, nmo_e, nocc_e)
        mo1_e = numpy.einsum('pj,cxji->cxpi', mo_coeff_e, mo1_e)
        e1_e = e1_e.reshape(-1, 3, nocc_e, nocc_e)

        hess = self.hess_elec(mo1_e = mo1_e, e1_e = e1_e)
        for i in range(len(self.mol.nuc)):
            mo_coeff_n = self.base.mf_nuc[i].mo_coeff
            mo_occ_n = self.base.mf_nuc[i].mo_occ
            nao_n, nmo_n = mo_coeff_n.shape
            mocc_n = mo_coeff_n[:,mo_occ_n>0]
            nocc_n = mocc_n.shape[1]

            mo1_n[i] = mo1_n[i].reshape(-1, 3, nmo_n, nocc_n)
            mo1_n[i] = numpy.einsum('pj,cxji->cxpi', mo_coeff_n, mo1_n[i])
            e1_n[i] = e1_n[i].reshape(-1, 3, nocc_n, nocc_n)
            f1[i] = f1[i].reshape(-1, 3, 3)

            hess += self.hess_elec_nuc1(i, mo1_n[i], mo1_e)
            hess += self.hess_elec_nuc2(i)
            hess += self.hess_nuc1(i, mo1_n[i], e1_n[i], f1[i])
            hess += self.hess_nuc2(i)

        hess += self.hess_nuc_nuc1()
        hess += self.hess_nuc_nuc2()

        print(hess)
        return hess


from pyscf.neo import CDFT
CDFT.Hessian = lib.class_as_method(Hessian)


