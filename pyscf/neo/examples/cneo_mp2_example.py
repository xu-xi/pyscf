#!/usr/bin/env python

from pyscf import neo


def build_mol():
    # HF with H treated as a quantum nucleus
    mol = neo.Mole()
    mol.build(
        atom='H 0.000000 0.000000 0.000000; F 0.000000 0.000000 0.900000',
        unit='Angstrom',
        basis='sto-3g',
        nuc_basis='pb4d',
        quantum_nuc=[0],
        charge=0,
        spin=0,
        verbose=4,
    )
    return mol


def main():
    mol = build_mol()

    # CNEO-HF reference (use CDFT with xc='hf')
    mf = neo.CDFT(mol, xc='hf')
    mf.conv_tol = 1e-10
    mf.conv_tol_grad = 1e-8
    mf.kernel()

    # CNEO-MP2(ee): electron-electron correlation only
    mp2_ee = neo.MP2(mf, with_ep=False)
    mp2_ee.kernel()
    print('--- CNEO-MP2(ee) ---')
    print('E(HF)        = %.12f' % mp2_ee.e_hf)
    print('E_corr(ee)   = %.12f' % mp2_ee.e_corr_ee)
    print('E_tot        = %.12f' % mp2_ee.e_tot)

    # Full CNEO-MP2: includes electron-electron and electron-proton correlation
    mp2_full = neo.MP2(mf, with_ep=True)
    mp2_full.kernel()
    print('--- CNEO-MP2 (full) ---')
    print('E_corr(ee)   = %.12f' % mp2_full.e_corr_ee)
    print('E_corr(ep)   = %.12f' % mp2_full.e_corr_ep)
    print('E_tot        = %.12f' % mp2_full.e_tot)


if __name__ == '__main__':
    main()
