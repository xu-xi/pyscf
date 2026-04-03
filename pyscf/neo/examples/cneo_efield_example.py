#!/usr/bin/env python

from ase import Atoms
from ase.optimize import BFGS
from ase.vibrations import Infrared

from pyscf.neo import Pyscf_NEO


def build_atoms():
    atoms = Atoms(
        symbols=['O', 'H', 'H'],
        positions=[
            (0.00000000, 0.00000000, 0.11697300),
            (0.00000000, 0.76339900, -0.46789400),
            (0.00000000, -0.76339900, -0.46789400),
        ],
    )
    atoms.calc = Pyscf_NEO(
        basis='aug-ccpvtz',
        xc='b3lyp',
        charge=0,
        efield=(0.0, 0.0, -0.02),
    )
    return atoms


def main():
    atoms = build_atoms()

    opt = BFGS(atoms, trajectory='H2O-opt-cneo.traj')
    opt.run(fmax=0.005)

    vib = Infrared(atoms, name='H2O-vib-cneo')
    vib.run()
    vib.combine()
    vib.summary()
    vib.write_spectra(out='H2O-cneo-ir.dat', start=0, end=5000)
    vib.write_jmol()
    vib.write_mode()


if __name__ == '__main__':
    main()
