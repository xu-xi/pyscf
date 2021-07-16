#!/usr/bin/env python

import numpy
import unittest
from pyscf import neo

mol = neo.Mole()
mol.build(atom='''H 0 0 0; C 0 0 1.064; N 0 0 2.220''', basis='ccpvdz', quantum_nuc=[0])
mf = neo.HF(mol)
energy = mf.scf()

class KnownValues(unittest.TestCase):
    def test_scf(self):
        self.assertAlmostEqual(energy, -92.8437063380073, 9)

if __name__ == "__main__":
    print("Full Tests for neo.hf")
    unittest.main()
