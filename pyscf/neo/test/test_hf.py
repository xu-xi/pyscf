#!/usr/bin/env python

import numpy
import unittest
from pyscf import neo

'''
PB4-D basis for proton and even-tempered basis for other heavy nuclei
Electronic xc: b3lyp
'''

class KnownValues(unittest.TestCase):
    def test_scf_H(self):
        mol = neo.Mole()
        mol.build(atom='''H 0 0 0; C 0 0 1.064; N 0 0 2.220''', basis='ccpvdz', quantum_nuc=[0])
        mf = neo.HF(mol)
        energy = mf.scf()
        self.assertAlmostEqual(energy, -92.8437063572664, 8)

    def test_scf2(self):
        mol = neo.Mole()
        mol.build(atom='''H 0 0 0; C 0 0 1.064; N 0 0 2.220''', basis='ccpvdz', quantum_nuc=[0])
        mf = neo.HF(mol)
        energy = mf.scf2()
        self.assertAlmostEqual(energy, -92.8437063572664, 8)

    def test_scf_df(self):
        mol = neo.Mole()
        mol.build(atom='''H 0 0 0; C 0 0 1.064; N 0 0 2.220''', basis='ccpvdz', quantum_nuc=[0])
        mf = neo.HF(mol)
        mf.with_df = True
        energy = mf.scf()
        self.assertAlmostEqual(energy, -92.8437061030512, 7)

    def test_scf_full_quantum(self):
        mol = neo.Mole()
        mol.build(atom='''H 0 0 0; C 0 0 1.064; N 0 0 2.220''', basis='ccpvdz', quantum_nuc=[0,1,2])
        mf = neo.HF(mol)
        energy = mf.scf()
        self.assertAlmostEqual(energy, -91.6090279376862, 8)


if __name__ == "__main__":
    print("Full Tests for neo.hf")
    unittest.main()
