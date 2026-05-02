#!/usr/bin/env python
# Copyright 2014-2026 The PySCF Developers. All Rights Reserved.
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

"""
CNEO-MP2 and CNEO-MP2(ee).

CNEO-MP2(ee) adds only the electron-electron MP2 correction on top of the
converged (C)NEO-HF reference. The full CNEO-MP2 additionally
includes electron-proton dynamic correlation.
"""

import numpy
from pyscf import gto
from pyscf import lib
from pyscf.lib import logger
from pyscf import ao2mo
from pyscf import mp
from pyscf import scf


def as_scanner(mp2_obj):
    """Generate a scanner for CNEO-MP2 PES."""
    if isinstance(mp2_obj, lib.SinglePointScanner):
        return mp2_obj

    logger.info(mp2_obj, 'Create scanner for %s', mp2_obj.__class__)
    name = mp2_obj.__class__.__name__ + CNEOMP2_Scanner.__name_mixin__
    return lib.set_class(CNEOMP2_Scanner(mp2_obj),
                         (CNEOMP2_Scanner, mp2_obj.__class__), name)


class CNEOMP2_Scanner(lib.SinglePointScanner):
    def __init__(self, mp2_obj):
        self.__dict__.update(mp2_obj.__dict__)
        self.base = mp2_obj.base.as_scanner()

    def __call__(self, mol_or_geom, **kwargs):
        if isinstance(mol_or_geom, gto.MoleBase):
            mol = mol_or_geom
        else:
            mol = self.mol.set_geom_(mol_or_geom, inplace=False)

        self.reset(mol)
        self.e_hf = self.base(mol, **kwargs)
        self.mol = self.base.mol

        self.mp_e = mp.MP2(self.base.components['e'])
        self.mp_e.max_memory = self.max_memory
        self.mp_e.verbose = self.verbose
        self.mp_e.stdout = self.stdout
        self.e_corr_ee, self.t2 = self.mp_e.kernel()
        self.e_corr_ee = float(self.e_corr_ee)
        self.e_corr_ep = 0.0
        self.t2_ep = None

        if self.with_ep:
            self.e_corr_ep, self.t2_ep = self._compute_ep_corr(with_t2=True)

        self.e_corr = float(self.e_corr_ee + self.e_corr_ep)
        self.e_tot = self.e_hf + self.e_corr
        return self.e_tot

    @property
    def converged(self):
        return self.base.converged


class MP2(lib.StreamObject):
    """CNEO-MP2 and CNEO-MP2(ee).

    Parameters
    ----------
    mf : pyscf.neo.hf.HF or pyscf.neo.cdft.CDFT
        Converged (C)NEO reference object.
    with_ep : bool
        If True, include explicit electron-proton correlation (CNEO-MP2).
        If False, compute the simplified CNEO-MP2(ee).
    mp2_grad_slow : bool
        If True, use the slow direct CNEO-CPHF implementation for
        gradients. If False, use the Z-vector gradient implementation for
        CNEO-MP2(ee), and for full CNEO-MP2 when there are no frozen
        electronic orbitals. The default is False.
    """
    _keys = {'base', 'mol', 'verbose', 'stdout', 'max_memory',
             'with_ep', 'mp_e', 'e_hf', 'e_corr', 'e_tot', 't2',
             'e_corr_ee', 'e_corr_ep', 't2_ep', 'mp2_grad_slow'}

    def __init__(self, mf, with_ep=False, mp2_grad_slow=False):
        if not hasattr(mf, 'components') or 'e' not in mf.components:
            raise TypeError('CNEO-MP2 requires a (C)NEO reference object.')

        self.base = mf
        self.mol = mf.mol
        self.verbose = self.mol.verbose
        self.stdout = self.mol.stdout
        self.max_memory = mf.max_memory
        self.with_ep = bool(with_ep)
        self.mp2_grad_slow = bool(mp2_grad_slow)

        if isinstance(mf.components['e'], scf.uhf.UHF):
            raise NotImplementedError('CNEO-MP2 currently supports RHF electron reference only')

        self.mp_e = None
        self.e_hf = None
        self.e_corr = None
        self.e_corr_ee = None
        self.e_corr_ep = None
        self.e_tot = None
        self.t2 = None
        self.t2_ep = None

    def dump_flags(self, verbose=None):
        log = logger.new_logger(self, verbose)
        log.info('')
        log.info('******** %s ********', self.__class__)
        log.info('Reference = %s', self.base.__class__.__name__)
        log.info('Model     = %s', 'CNEO-MP2' if self.with_ep else 'CNEO-MP2(ee)')
        log.info('max_memory %d MB (current use %d MB)',
                 self.max_memory, lib.current_memory()[0])
        return self

    def reset(self, mol=None):
        if mol is not None:
            self.mol = mol
        self.base.reset(mol)
        # A reset changes the geometry. Force the CNEO reference to be
        # recomputed on the next kernel call instead of reusing stale
        # converged orbitals/energies from the previous geometry.
        self.base.converged = False
        for comp in self.base.components.values():
            comp.converged = False
        self.mp_e = None
        self.e_hf = None
        self.e_corr = None
        self.e_corr_ee = None
        self.e_corr_ep = None
        self.e_tot = None
        self.t2 = None
        self.t2_ep = None
        return self

    def kernel(self, with_t2=True):
        if not self.base.converged:
            self.base.kernel()

        if self.verbose >= logger.INFO:
            self.dump_flags()

        self.e_hf = float(self.base.e_tot)

        self.mp_e = mp.MP2(self.base.components['e'])
        self.mp_e.max_memory = self.max_memory
        self.mp_e.verbose = self.verbose
        self.mp_e.stdout = self.stdout
        self.e_corr_ee, self.t2 = self.mp_e.kernel(with_t2=with_t2)
        self.e_corr_ee = float(self.e_corr_ee)

        self.e_corr_ep = 0.0
        self.t2_ep = None
        if self.with_ep:
            self.e_corr_ep, self.t2_ep = self._compute_ep_corr(with_t2=with_t2)

        self.e_corr = float(self.e_corr_ee + self.e_corr_ep)
        self.e_tot = self.e_hf + self.e_corr

        if self.with_ep:
            logger.note(self, 'E(CNEO-MP2) = %.15g  E_corr = %.15g  '
                        'E_corr(ee) = %.15g  E_corr(ep) = %.15g',
                        self.e_tot, self.e_corr, self.e_corr_ee, self.e_corr_ep)
        else:
            logger.note(self, 'E(CNEO-MP2(ee)) = %.15g  E_corr = %.15g',
                        self.e_tot, self.e_corr)
        return self.e_corr, self.t2

    as_scanner = as_scanner

    def nuc_grad_method(self):
        from pyscf.neo import mp2_grad
        return mp2_grad.Gradients(self)

    @staticmethod
    def _ep_ovov(mol_e, mol_n, ce, cn, occidx_e, viridx_e, occidx_n, viridx_n):
        idx_occ_e = numpy.where(occidx_e)[0]
        idx_vir_e = numpy.where(viridx_e)[0]
        idx_occ_n = numpy.where(occidx_n)[0]
        idx_vir_n = numpy.where(viridx_n)[0]

        nocc_e = idx_occ_e.size
        nvir_e = idx_vir_e.size
        nocc_n = idx_occ_n.size
        nvir_n = idx_vir_n.size

        nao_e = mol_e.nao_nr()
        nao_n = mol_n.nao_nr()
        nao_tot = nao_e + nao_n

        co_e = numpy.zeros((nao_tot, nocc_e))
        cv_e = numpy.zeros((nao_tot, nvir_e))
        co_n = numpy.zeros((nao_tot, nocc_n))
        cv_n = numpy.zeros((nao_tot, nvir_n))

        co_e[:nao_e, :] = ce[:, idx_occ_e]
        cv_e[:nao_e, :] = ce[:, idx_vir_e]
        co_n[nao_e:, :] = cn[:, idx_occ_n]
        cv_n[nao_e:, :] = cn[:, idx_vir_n]

        mol_tot = mol_e + mol_n
        eri = mol_tot.intor('int2e', aosym='s4')
        eri_ovov = ao2mo.incore.general(eri, (co_e, cv_e, co_n, cv_n),
                                        compact=False)
        return eri_ovov.reshape(nocc_e, nvir_e, nocc_n, nvir_n)

    def _compute_ep_corr(self, with_t2=True):
        mf = self.base
        comp_e = mf.components['e']

        mo_occ_e = numpy.asarray(mf.mo_occ['e'])
        if mo_occ_e.ndim != 1:
            raise NotImplementedError('CNEO-MP2 ep correlation requires restricted '
                                      'electronic references')

        occidx_e = mo_occ_e > 0
        viridx_e = ~occidx_e
        if not occidx_e.any() or not viridx_e.any():
            return 0.0, {} if with_t2 else None

        eps_e = mf.mo_energy['e']
        eps_i = eps_e[occidx_e]
        eps_a = eps_e[viridx_e]
        ce = mf.mo_coeff['e']

        e_corr_ep = 0.0
        t2_ep = {} if with_t2 else None

        for t, comp_n in mf.components.items():
            if not t.startswith('n'):
                continue

            mo_occ_n = numpy.asarray(mf.mo_occ[t])
            if mo_occ_n.ndim != 1:
                raise NotImplementedError('CNEO-MP2 ep correlation requires restricted '
                                          'nuclear references')

            occidx_n = mo_occ_n > 0
            viridx_n = ~occidx_n
            if not occidx_n.any() or not viridx_n.any():
                continue

            eps_n = mf.mo_energy[t]
            eps_I = eps_n[occidx_n]
            eps_A = eps_n[viridx_n]
            cn = mf.mo_coeff[t]

            eri_ovov = self._ep_ovov(comp_e.mol, comp_n.mol, ce, cn,
                                     occidx_e, viridx_e, occidx_n, viridx_n)
            charge_prod = comp_e.charge * comp_n.charge
            g = eri_ovov * charge_prod

            denom = (eps_i[:, None, None, None]
                     + eps_I[None, None, :, None]
                     - eps_a[None, :, None, None]
                     - eps_A[None, None, None, :])

            e_corr_ep += numpy.sum(2 * g * g / denom)

            if with_t2:
                t2_ep[t] = g / denom

        return e_corr_ep, t2_ep


class CNEOMP2(MP2):
    """Convenience wrapper for the full CNEO-MP2 with explicit ep correlation."""
    def __init__(self, mf, **kwargs):
        super().__init__(mf, with_ep=True, **kwargs)


from pyscf.neo import hf, ks, cdft  # noqa: E402
hf.HF.MP2 = lib.class_as_method(MP2)
ks.KS.MP2 = lib.class_as_method(MP2)
cdft.CDFT.MP2 = lib.class_as_method(MP2)
hf.HF.CNEOMP2 = lib.class_as_method(CNEOMP2)
ks.KS.CNEOMP2 = lib.class_as_method(CNEOMP2)
cdft.CDFT.CNEOMP2 = lib.class_as_method(CNEOMP2)
