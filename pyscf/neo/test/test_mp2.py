#!/usr/bin/env python

import unittest
import numpy
from pyscf import lib, neo
from pyscf.neo.mp2 import MP2


BOHR = lib.param.BOHR
STEP = 1e-3
GRAD_TOL = 1e-5


def build_mol(geom, charge, spin, quantum_nuc, basis_e='sto-3g', nuc_basis='pb4d'):
    mol = neo.Mole()
    mol.build(atom=geom,
              unit='Angstrom',
              basis=basis_e,
              nuc_basis=nuc_basis,
              quantum_nuc=quantum_nuc,
              charge=charge,
              spin=spin)
    mol.verbose = 0
    return mol


def build_mp2(mol, with_ep=False, mp2_grad_slow=True):
    mf = neo.CDFT(mol, xc='hf')
    mf.conv_tol = 1e-10
    mf.conv_tol_grad = 1e-8
    mf.verbose = 0
    mf.kernel()

    mp2obj = MP2(mf, with_ep=with_ep, mp2_grad_slow=mp2_grad_slow)
    mp2obj.verbose = 0
    mp2obj.kernel()
    return mp2obj


def analytic_grad(mp2ee):
    gobj = mp2ee.nuc_grad_method()
    gobj.verbose = 0
    return gobj.kernel()


def numeric_grad(mp2ee, step=STEP):
    mol = mp2ee.mol
    coords = mol.atom_coords(unit='Angstrom')
    syms = [mol.atom_symbol(i) for i in range(mol.natm)]
    scanner = mp2ee.as_scanner()
    g_num = numpy.zeros((mol.natm, 3))

    for ia in range(mol.natm):
        for comp in range(3):
            coords_p = coords.copy()
            coords_m = coords.copy()
            coords_p[ia, comp] += step
            coords_m[ia, comp] -= step
            geom_p = '; '.join(
                f'{syms[i]} {coords_p[i,0]:.12f} {coords_p[i,1]:.12f} {coords_p[i,2]:.12f}'
                for i in range(mol.natm)
            )
            geom_m = '; '.join(
                f'{syms[i]} {coords_m[i,0]:.12f} {coords_m[i,1]:.12f} {coords_m[i,2]:.12f}'
                for i in range(mol.natm)
            )
            e_p = scanner(geom_p)
            e_m = scanner(geom_m)
            g_num[ia, comp] = (e_p - e_m) / (2 * step) * BOHR
    return g_num


class KnownValues(unittest.TestCase):
    def check_mp2_grad(self, geom, charge, spin, quantum_nuc,
                       basis_e='ccpvdz', nuc_basis='pb4d',
                       with_ep=False, mp2_grad_slow=True,
                       step=STEP, tol=GRAD_TOL):
        mol = build_mol(geom, charge, spin, quantum_nuc, basis_e, nuc_basis)
        mp2obj = build_mp2(mol, with_ep=with_ep, mp2_grad_slow=mp2_grad_slow)
        g_ana = analytic_grad(mp2obj)
        g_num = numeric_grad(mp2obj, step=step)
        diff = g_ana - g_num

        self.assertLess(numpy.max(numpy.abs(diff)), tol)

    def test_grad_h2o(self):
        self.check_mp2_grad(
            geom='O 0.000000 0.000000 0.000000; '
                 'H 0.000000 -0.757000 0.587000; '
                 'H 0.000000 0.757000 0.587000',
            charge=0,
            spin=0,
            quantum_nuc=[1, 2],
        )

    def test_grad_fhf(self):
        self.check_mp2_grad(
            geom='F 0.000000 0.000000 -1.000000; '
                 'H 0.000000 0.000000 0.000000; '
                 'F 0.000000 0.000000 1.000000',
            charge=-1,
            spin=0,
            quantum_nuc=[1],
        )

    def test_grad_hf(self):
        self.check_mp2_grad(
            geom='H 0.000000 0.000000 0.000000; F 0.000000 0.000000 0.900000',
            charge=0,
            spin=0,
            quantum_nuc=[0],
        )

    def test_grad_hf_z_vector(self):
        self.check_mp2_grad(
            geom='H 0.000000 0.000000 0.000000; F 0.000000 0.000000 0.900000',
            charge=0,
            spin=0,
            quantum_nuc=[0],
            mp2_grad_slow=False,
            tol=1e-4,
        )

    def test_grad_hf_full_ep(self):
        self.check_mp2_grad(
            geom='H 0.000000 0.000000 0.000000; F 0.000000 0.000000 0.900000',
            charge=0,
            spin=0,
            quantum_nuc=[0],
            with_ep=True,
        )

    def test_grad_h2o_full_ep(self):
        self.check_mp2_grad(
            geom='O 0.000000 0.000000 0.000000; '
                 'H 0.000000 -0.757000 0.587000; '
                 'H 0.000000 0.757000 0.587000',
            charge=0,
            spin=0,
            quantum_nuc=[1, 2],
            with_ep=True,
        )

    def test_grad_hcn_full_ep(self):
        self.check_mp2_grad(
            geom='H 0.000000 0.000000 0.000000; '
                 'C 0.000000 0.000000 1.064000; '
                 'N 0.000000 0.000000 2.220000',
            charge=0,
            spin=0,
            quantum_nuc=[0],
            with_ep=True,
        )

    def test_reset_recomputes_reference(self):
        mol0 = build_mol(
            geom='H 0.000000 0.000000 -0.018267995234; '
                 'F 0.000000 0.000000 0.917738818023',
            charge=0,
            spin=0,
            quantum_nuc=[0],
        )
        coords = mol0.atom_coords()
        coords[0, 0] += 1e-3  # Bohr
        mol1 = mol0.set_geom_(coords, unit='Bohr', inplace=False)

        mp2_fresh = build_mp2(mol1)
        g_fresh = analytic_grad(mp2_fresh)

        mp2_reuse = build_mp2(mol0)
        mp2_reuse.reset(mol1)
        mp2_reuse.kernel()
        g_reuse = analytic_grad(mp2_reuse)

        self.assertAlmostEqual(mp2_reuse.e_tot, mp2_fresh.e_tot, places=10)
        self.assertLess(numpy.max(numpy.abs(g_reuse - g_fresh)), 1e-8)


if __name__ == '__main__':
    print('Full Tests for neo.mp2')
    unittest.main()
