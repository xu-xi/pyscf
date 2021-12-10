#!/usr/bin/env python

'''
Density fitting for NEO
'''

import copy
import numpy
import scipy
from pyscf import gto
from pyscf import df

def get_eri_ne_df(mf_neo):
    ''' get Coulomb integral between quantum nucleus and electrons from density fitting '''

    df_eri = [None] * mf_neo.mol.nuc_num

    for i in range(mf_neo.mol.nuc_num):
        mole = mf_neo.mol.nuc[i]
        ia = mole.atom_index

        # set up the auxiliary basis
        alpha = 4* numpy.sqrt(2) * mf_neo.mol.mass[ia]
        beta = numpy.sqrt(2)
        n = 32
        basis = gto.expand_etbs([(0, n, alpha, beta), (1, n, alpha, beta), (2, n, alpha, beta), (3, n, alpha, beta)])

        # build auxmol
        auxmol = copy.copy(mole)
        auxmol.build(atom = mole.atom, basis={mole.atom_symbol(ia): basis},
            charge = mole.charge, cart = mole.cart, spin = mole.spin)

        # calculate 3c2e and 2c2e
        ints_3c2e_nuc = df.incore.aux_e2(mole, auxmol, intor='int3c2e')
        ints_3c2e_elec = df.incore.aux_e2(mf_neo.mol.elec, auxmol, intor='int3c2e')
        ints_2c2e = auxmol.intor('int2c2e')
        #w, _ = numpy.linalg.eig(ints_2c2e)
        #print(w)

        nao_e = mf_neo.mol.elec.nao
        nao_n = mole.nao
        naux = auxmol.nao

        # Compute the DF coefficients (df_coef) and the DF 2-electron (df_eri)
        df_coef = scipy.linalg.solve(ints_2c2e, ints_3c2e_nuc.reshape(nao_n*nao_n, naux).T)
        df_coef = df_coef.reshape(naux, nao_n, nao_n)

        df_eri[i] = numpy.einsum('ijP,Pkl->ijkl', ints_3c2e_elec, df_coef)

    return df_eri


def get_eri_nn_df(mf_neo, nuc):
    ''' get Coulomb integral between quantum nuclei from density fitting '''
    pass
