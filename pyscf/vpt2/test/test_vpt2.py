#!/usr/bin/env python
# Copyright 2026 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import unittest
import numpy as np

from pyscf import gto, scf, dft, vpt2


class KnownValues(unittest.TestCase):
    def test_vpt2_h2_diag(self):
        mol = gto.M(atom='H 0 0 0; H 0 0 0.74', basis='sto-3g', verbose=0)
        mf = scf.RHF(mol).run(conv_tol=1e-12)
        res = vpt2.kernel(mf, displacement=2e-2, quartic='diag')

        self.assertEqual(res['freq_anharm_wavenumber'].size, 1)
        self.assertEqual(res['phi3'].shape, (1, 1, 1))
        self.assertEqual(res['phi4'].shape, (1, 1))
        self.assertTrue(np.isfinite(res['freq_anharm_wavenumber'][0]))
        self.assertGreater(res['freq_anharm_wavenumber'][0], 0.0)
        self.assertEqual(res['mode_index'][0], 0)

    def test_vpt2_modes_filter(self):
        mol = gto.M(atom='O 0 0 0; H 0 .757 .587; H 0 -.757 .587',
                    basis='sto-3g', verbose=0)
        mf = scf.RHF(mol).run(conv_tol=1e-12)
        res = vpt2.kernel(mf, displacement=2e-2, quartic='none', modes=[0])

        self.assertEqual(res['freq_anharm_wavenumber'].size, 1)
        self.assertIsNone(res['phi4'])
        self.assertEqual(res['mode_index'][0], 0)

    #@unittest.skipUnless(os.getenv('PYSCF_VPT2_LONG') == '1',
    #                     'set PYSCF_VPT2_LONG=1 to run the B3LYP/aug-cc-pVTZ test')
    def test_vpt2_h2o_b3lyp_augccpvtz_high_cost(self):
        mol = gto.M(atom='O 0 0 0; H 0 .757 .587; H 0 -.757 .587',
                    basis='aug-cc-pvtz', verbose=0)
        mf = dft.RKS(mol)
        mf.xc = 'b3lyp'
        mf.conv_tol = 1e-12
        mf.conv_tol_cpscf = 1e-9
        mf.kernel()
        res = vpt2.kernel(mf, displacement=2e-2, quartic='none', modes=[0])
        print(res)

        self.assertEqual(res['freq_anharm_wavenumber'].size, 1)
        self.assertTrue(np.isfinite(res['freq_anharm_wavenumber'][0]))
        self.assertGreater(res['freq_anharm_wavenumber'][0], 0.0)


if __name__ == "__main__":
    print("Full Tests for VPT2")
    unittest.main()
