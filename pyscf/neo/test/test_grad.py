#!/usr/bin/env python

import numpy
import unittest
from pyscf import neo

mol = neo.Mole()
mol.build(atom='''H 0 0 0; C 0 0 1.064; N 0 0 2.220''', basis='ccpvdz', quantum_nuc=[0])

class KnownValues(unittest.TestCase):
    def test_grad_cdft(self):
        mf = neo.CDFT(mol)
        mf.scf()
        grad = [[0.0000000000, 0.0000000000, 0.0249735118],
            [0.0000000000, 0.0000000000, -0.0197857809],
            [0.0000000000, 0.0000000000, -0.0051923531]]
        g = neo.Gradients(mf)
        self.assertTrue(numpy.allclose(g.kernel(), grad))



if __name__ == "__main__":
    print("Full Tests for neo.ks")
    unittest.main()
