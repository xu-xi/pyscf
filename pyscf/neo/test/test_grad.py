#!/usr/bin/env python

import numpy
import unittest
from pyscf import neo



class KnownValues(unittest.TestCase):
    def test_grad_cdft(self):
        mol = neo.Mole()
        mol.build(atom='''H 0 0 0; F 0 0 0.94''', basis='ccpvdz', quantum_nuc=[0])
        mf = neo.CDFT(mol)
        mf.scf()
        g = neo.Gradients(mf)
        grad = g.kernel()
        self.assertAlmostEqual(grad[0,-1], 0.0051324194, 6)

    def test_grad_cdft2(self):
        mol = neo.Mole()
        mol.build(atom='''H 0 0 0; F 0 0 0.94''', basis='ccpvdz', quantum_nuc=[0,1])
        mf = neo.CDFT(mol)
        mf.scf()
        g = neo.Gradients(mf)
        grad = g.kernel()
        self.assertAlmostEqual(grad[0,-1], 0.0043045068, 6)


if __name__ == "__main__":
    print("Full Tests for neo.ks")
    unittest.main()
