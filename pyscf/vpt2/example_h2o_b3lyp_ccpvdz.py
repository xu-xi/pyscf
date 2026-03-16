#!/usr/bin/env python
# Example: H2O VPT2 at DFT/B3LYP with cc-pVDZ

from pyscf import gto, dft, vpt2


def main():
    # Geometry optimized at B3LYP/cc-pVDZ (see example workflow).
    mol = gto.M(
        atom='''
        O  0.00000000 -0.00000000 -0.01171034
        H  0.00000000  0.75683862  0.59285517
        H  0.00000000 -0.75683862  0.59285517
        ''',
        basis='cc-pvdz',
        verbose=4,
    )

    mf = dft.RKS(mol)
    mf.xc = 'b3lyp'
    mf.conv_tol = 1e-12
    mf.conv_tol_cpscf = 1e-9
    mf.kernel()

    # Full semi-diagonal quartic VPT2 (more expensive).
    # For faster runs, use quartic='none' or limit modes (e.g., modes=[0]).
    res = vpt2.kernel(mf, displacement=2e-2, quartic='semi')

    print('Harmonic frequencies (cm^-1):')
    print(res['freq_wavenumber'])
    print('VPT2 anharmonic frequencies (cm^-1):')
    print(res['freq_anharm_wavenumber'])


if __name__ == '__main__':
    main()
