#!/usr/bin/env python

'''
Density fitting for NEO
'''

import copy
import numpy
import scipy
from pyscf import gto
from pyscf import df

def make_auxmol(mol):
    'Generate a fake Mole object for quantum nucleus using density fitting auxbasis'
    # TODO: test the basis for heavy nuclei
    # set up the auxiliary basis
    mass = mol.atom_mass_list(isotope_avg=True)
    ia = mol.atom_index

    alpha = 4* numpy.sqrt(2) * mass[ia] *2 # NOTE: alpha is crucial for the precision of df
    beta = numpy.sqrt(2)
    n = 8
    basis = gto.expand_etbs([(0, n, alpha, beta), (1, n, alpha, beta), (2, n, alpha, beta), (3, n, alpha, beta)])

    # build auxmol
    auxmol = gto.Mole()
    auxmol.build(atom = mol.atom, basis = {mol.atom_symbol(ia): basis},
                 charge = mol.charge, cart = mol.cart, spin = mol.spin)


    return auxmol

def get_eri_ne_df(mf_neo):
    '''
    get coefficents for density fitting of quantum nuclei
    and 2c2e-type integral between quantum nucleus and electrons
    '''

    df_coef = [None] * mf_neo.mol.nuc_num
    ints = [None] * mf_neo.mol.nuc_num

    for i in range(mf_neo.mol.nuc_num):
        mol = mf_neo.mol.nuc[i]
        ia = mol.atom_index

        auxmol = make_auxmol(mol)

        # calculate 3c2e and 2c2e
        ints_3c2e_nuc = df.incore.aux_e2(mol, auxmol, intor='int3c2e') # TODO: add aosym 's2ij'
        ints_2c2e = auxmol.intor('int2c2e', hermi=1)
        #w, _ = numpy.linalg.eig(ints_2c2e)
        # print(w)

        nao_e = mf_neo.mol.elec.nao
        nao_n = mol.nao
        naux = auxmol.nao
        #print(naux)

        # Compute the DF coefficients (df_coef) and the DF 2-electron (df_eri)
        coef = scipy.linalg.solve(ints_2c2e, ints_3c2e_nuc.reshape(nao_n*nao_n, naux).T, assume_a='sym')
        df_coef[i] = coef.reshape(naux, nao_n, nao_n)

        ints[i] = df.incore.aux_e2(mf_neo.mol.elec, auxmol, intor='int3c2e')

        #df_eri[i] = numpy.einsum('ijP,Pkl->ijkl', ints[i], df_coef[i])

    return df_coef, ints

def get_eri_nn_df(mf_neo, nuc):
    ''' get 2c2e-type integral between quantum nuclei from density fitting '''
    ia = nuc.atom_index

    auxmol = make_auxmol(nuc)
    ints = [None] * mf_neo.mol.nuc_num

    for j in range(mf_neo.mol.nuc_num):
        nuc2 = mf_neo.mol.nuc[j]
        ja = nuc2.atom_index
        if ja != ia:
            auxmol2 = make_auxmol(nuc2)
            ints[j] = df.incore.aux_e2(auxmol, auxmol2, intor='int2c2e')

    return ints
