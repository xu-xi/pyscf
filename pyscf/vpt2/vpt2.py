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

'''
VPT2 (second-order vibrational perturbation theory).

This module estimates anharmonic vibrational frequencies from cubic and
semi-diagonal quartic force constants obtained by finite-difference of
analytical Hessians along harmonic normal modes.

Limitations
-----------
- Requires analytical Hessian on displaced geometries.
- Uses fixed harmonic normal modes (no Duschinsky rotation).
- Ignores resonances; near-zero denominators are skipped.
'''

from __future__ import annotations

import numpy as np

from pyscf import lib
from pyscf.data import nist
from pyscf.hessian import thermo
from pyscf.lib import logger


def _au2hz():
    # For mass in amu (as used in pyscf.hessian.thermo.harmonic_analysis)
    return (nist.HARTREE2J / (nist.ATOMIC_MASS * nist.BOHR_SI**2))**.5 / (2 * np.pi)


def _hess_cart_to_matrix(hess):
    natm = hess.shape[0]
    return hess.transpose(0, 2, 1, 3).reshape(natm * 3, natm * 3)


def _get_scan(method):
    if isinstance(method, lib.SinglePointScanner):
        return method
    if hasattr(method, 'as_scanner'):
        return method.as_scanner()
    return method


def _run_scan(scan, mol):
    if isinstance(scan, lib.SinglePointScanner):
        scan(mol)
        return
    # Fall back to a direct run
    if hasattr(scan, 'run'):
        scan.run()
    else:
        scan.kernel()


def _compute_hessian(scan, mol):
    _run_scan(scan, mol)
    if not hasattr(scan, 'Hessian'):
        raise RuntimeError('Hessian is not available for %s' % scan.__class__.__name__)
    return scan.Hessian().kernel()


def _apply_q_ops(state, modes, omega):
    states = {state: 1.0}
    for m in modes:
        w = omega[m]
        new_states = {}
        for st, coeff in states.items():
            n = st[m]
            if n > 0:
                st_low = list(st)
                st_low[m] = n - 1
                st_low = tuple(st_low)
                val = coeff * np.sqrt(n / (2 * w))
                new_states[st_low] = new_states.get(st_low, 0.0) + val
            st_high = list(st)
            st_high[m] = n + 1
            st_high = tuple(st_high)
            val = coeff * np.sqrt((n + 1) / (2 * w))
            new_states[st_high] = new_states.get(st_high, 0.0) + val
        states = new_states
    return states


def _harmonic_energy(state, omega):
    n = np.asarray(state, dtype=float)
    return np.dot(omega, n + 0.5)


def _q2_expect(n, omega):
    return (n + 0.5) / omega


def _q4_expect(n, omega):
    return (3.0 + 6.0 * n + 6.0 * n * n) / (4.0 * omega * omega)


def _quartic_correction(state, omega, phi4):
    if phi4 is None:
        return 0.0
    nmode = len(omega)
    n = np.asarray(state, dtype=float)
    e1 = 0.0
    # Diagonal terms (iiii)
    for i in range(nmode):
        e1 += phi4[i, i] * _q4_expect(n[i], omega[i]) / 24.0
    # Semi-diagonal terms (iijj), i < j
    for i in range(nmode):
        qi2 = _q2_expect(n[i], omega[i])
        for j in range(i + 1, nmode):
            e1 += phi4[i, j] * qi2 * _q2_expect(n[j], omega[j]) / 4.0
    return e1


def _cubic_correction(state, omega, phi3, resonance_tol, log):
    nmode = len(omega)
    amp = {}
    for i in range(nmode):
        for j in range(nmode):
            for k in range(nmode):
                phi = phi3[i, j, k]
                if abs(phi) < 1e-14:
                    continue
                for m_state, coeff in _apply_q_ops(state, (i, j, k), omega).items():
                    amp[m_state] = amp.get(m_state, 0.0) + (phi / 6.0) * coeff

    e0 = _harmonic_energy(state, omega)
    e2 = 0.0
    resonances = []
    for m_state, val in amp.items():
        em = _harmonic_energy(m_state, omega)
        denom = e0 - em
        if abs(denom) < resonance_tol:
            resonances.append((state, m_state, denom))
            continue
        e2 += (val * val) / denom
    if resonances:
        log.warn('VPT2 resonance: %d near-zero denominators skipped', len(resonances))
    return e2, resonances


def kernel(method, displacement=1e-2, mass=None, modes=None, quartic='semi',
           resonance_tol=1e-4, drop_imaginary=True, verbose=None):
    '''
    Run a VPT2 calculation.

    Args:
        method:
            A converged PySCF method object providing analytic Hessian.

    Kwargs:
        displacement:
            Displacement along mass-weighted normal coordinates (in sqrt(amu)*Bohr).
        mass:
            Optional list of atomic masses in amu. Default uses isotope-averaged masses.
        modes:
            Optional list of mode indices (0-based) to include after filtering
            imaginary modes.
        quartic:
            'semi' for semi-diagonal quartic (iijj), 'diag' for diagonal only,
            or 'none' to skip quartic corrections.
        resonance_tol:
            Threshold (a.u.) to skip near-zero denominators in VPT2.
        drop_imaginary:
            If True, drop imaginary frequencies before VPT2.
    '''
    scan = _get_scan(method)
    mol = scan.mol
    if verbose is None:
        verbose = mol.verbose
    log = logger.new_logger(mol, verbose)

    quartic = (quartic or 'semi').lower()
    if quartic not in ('semi', 'diag', 'none'):
        raise ValueError('quartic must be one of: semi, diag, none')

    orig_coords = mol.atom_coords()
    try:
        hess0 = _compute_hessian(scan, mol)
        if mass is None:
            mass = mol.atom_mass_list(isotope_avg=True)
        mass = np.asarray(mass, dtype=float)
        harm = thermo.harmonic_analysis(mol, hess0, mass=mass,
                                        exclude_trans=True, exclude_rot=True,
                                        imaginary_freq=True)
        # harm['freq_au'] is based on amu masses. Convert to true atomic units
        # (electron mass) for VPT2 formulas.
        freq_amu = harm['freq_au']
        mass_scale = np.sqrt(nist.AMU2AU)
        freq = freq_amu / mass_scale
        norm_mode = harm['norm_mode']
        reduced_mass = harm['reduced_mass']

        mode_index = np.arange(freq.size)
        if drop_imaginary:
            mask = (freq.real > 0) & (abs(freq.imag) < 1e-8)
            if not mask.all():
                log.warn('Imaginary modes detected; dropping %d modes',
                         np.count_nonzero(~mask))
            mode_index = mode_index[mask]
            freq_amu = freq_amu[mask]
            freq = freq.real[mask]
            norm_mode = norm_mode[mask]
            reduced_mass = reduced_mass[mask]
        else:
            if np.iscomplexobj(freq):
                log.warn('Complex frequencies detected; using real part')
            freq_amu = freq_amu.real
            freq = freq.real

        if modes is not None:
            mode_idx = np.asarray(modes, dtype=int)
            if mode_idx.size == 0:
                raise ValueError('modes cannot be empty')
            if mode_idx.min() < 0 or mode_idx.max() >= freq.size:
                raise ValueError('modes index out of range')
            mode_index = mode_index[mode_idx]
            freq_amu = freq_amu[mode_idx]
            freq = freq[mode_idx]
            norm_mode = norm_mode[mode_idx]
            reduced_mass = reduced_mass[mode_idx]

        if np.any(freq <= 0):
            raise RuntimeError('Non-positive harmonic frequencies found; VPT2 is not applicable')

        nmode = freq.size
        if nmode == 0:
            raise RuntimeError('No vibrational modes available for VPT2')

        bmat = norm_mode.reshape(nmode, -1).T  # (3N, nmode)
        h0_cart = _hess_cart_to_matrix(hess0)
        h0_q = bmat.T @ h0_cart @ bmat

        # Precompute Hessians for single displacements
        hq_plus = [None] * nmode
        hq_minus = [None] * nmode
        # Convert displacement from sqrt(amu)*Bohr to sqrt(me)*Bohr to match mass_au
        disp_vecs = [norm_mode[i].reshape(-1) * displacement for i in range(nmode)]

        for k in range(nmode):
            coords_p = orig_coords + disp_vecs[k].reshape(mol.natm, 3)
            mol.set_geom_(coords_p, unit='Bohr')
            hess_p = _compute_hessian(scan, mol)
            hq_plus[k] = bmat.T @ _hess_cart_to_matrix(hess_p) @ bmat

            coords_m = orig_coords - disp_vecs[k].reshape(mol.natm, 3)
            mol.set_geom_(coords_m, unit='Bohr')
            hess_m = _compute_hessian(scan, mol)
            hq_minus[k] = bmat.T @ _hess_cart_to_matrix(hess_m) @ bmat

        # Cubic force constants (fully symmetric)
        phi3 = np.zeros((nmode, nmode, nmode))
        for k in range(nmode):
            phi3[:, :, k] = (hq_plus[k] - hq_minus[k]) / (2.0 * displacement)
        phi3 = (phi3 + phi3.transpose(1, 0, 2) + phi3.transpose(2, 1, 0) +
                phi3.transpose(0, 2, 1) + phi3.transpose(1, 2, 0) +
                phi3.transpose(2, 0, 1)) / 6.0

        # Semi-diagonal quartic (iijj)
        phi4 = None
        if quartic != 'none':
            phi4 = np.zeros((nmode, nmode))
            for i in range(nmode):
                phi4[i, i] = (hq_plus[i][i, i] - 2.0 * h0_q[i, i] + hq_minus[i][i, i]) / (displacement**2)
            if quartic == 'semi':
                for i in range(nmode):
                    for j in range(i + 1, nmode):
                        disp_pp = disp_vecs[i] + disp_vecs[j]
                        disp_pm = disp_vecs[i] - disp_vecs[j]
                        disp_mp = -disp_vecs[i] + disp_vecs[j]
                        disp_mm = -disp_vecs[i] - disp_vecs[j]

                        coords = orig_coords + disp_pp.reshape(mol.natm, 3)
                        mol.set_geom_(coords, unit='Bohr')
                        hpp = bmat.T @ _hess_cart_to_matrix(_compute_hessian(scan, mol)) @ bmat

                        coords = orig_coords + disp_pm.reshape(mol.natm, 3)
                        mol.set_geom_(coords, unit='Bohr')
                        hpm = bmat.T @ _hess_cart_to_matrix(_compute_hessian(scan, mol)) @ bmat

                        coords = orig_coords + disp_mp.reshape(mol.natm, 3)
                        mol.set_geom_(coords, unit='Bohr')
                        hmp = bmat.T @ _hess_cart_to_matrix(_compute_hessian(scan, mol)) @ bmat

                        coords = orig_coords + disp_mm.reshape(mol.natm, 3)
                        mol.set_geom_(coords, unit='Bohr')
                        hmm = bmat.T @ _hess_cart_to_matrix(_compute_hessian(scan, mol)) @ bmat

                        val = 0.5 * (hpp[i, j] + hpp[j, i])
                        val -= 0.5 * (hpm[i, j] + hpm[j, i])
                        val -= 0.5 * (hmp[i, j] + hmp[j, i])
                        val += 0.5 * (hmm[i, j] + hmm[j, i])
                        phi = val / (4.0 * displacement**2)
                        phi4[i, j] = phi4[j, i] = phi

        # Convert force constants from amu-based coordinates to atomic units
        phi3 /= mass_scale**3
        if phi4 is not None:
            phi4 /= mass_scale**4

        # VPT2 energies
        omega = freq
        zpe_harm = 0.5 * omega.sum()

        state0 = tuple([0] * nmode)
        e0_1 = _quartic_correction(state0, omega, phi4)
        e0_2, res0 = _cubic_correction(state0, omega, phi3, resonance_tol, log)
        e0_vpt2 = _harmonic_energy(state0, omega) + e0_1 + e0_2

        freq_anharm = np.zeros(nmode)
        resonances = {'ground': res0, 'fundamentals': []}
        for i in range(nmode):
            st = [0] * nmode
            st[i] = 1
            st = tuple(st)
            e1 = _quartic_correction(st, omega, phi4)
            e2, res = _cubic_correction(st, omega, phi3, resonance_tol, log)
            etot = _harmonic_energy(st, omega) + e1 + e2
            freq_anharm[i] = etot - e0_vpt2
            resonances['fundamentals'].append(res)

        au2hz = _au2hz() * mass_scale
        freq_cm = omega * au2hz / nist.LIGHT_SPEED_SI * 1e-2
        freq_anharm_cm = freq_anharm * au2hz / nist.LIGHT_SPEED_SI * 1e-2

        results = {
            'freq_au': omega,
            'freq_amu': freq_amu,
            'freq_wavenumber': freq_cm,
            'freq_anharm_au': freq_anharm,
            'freq_anharm_wavenumber': freq_anharm_cm,
            'zpe_harm_au': zpe_harm,
            'zpe_anharm_au': e0_vpt2,
            'phi3': phi3,
            'phi4': phi4,
            'norm_mode': norm_mode,
            'reduced_mass': reduced_mass,
            'mode_index': mode_index,
            'resonances': resonances,
            'settings': {
                'displacement': displacement,
                'displacement_au': displacement * mass_scale,
                'quartic': quartic,
                'resonance_tol': resonance_tol,
            },
            'harmonic': harm,
        }
        return results
    finally:
        mol.set_geom_(orig_coords, unit='Bohr')


class VPT2(lib.StreamObject):
    '''VPT2 driver.'''
    _keys = {
        'base', 'mol', 'verbose', 'stdout', 'displacement', 'mass',
        'modes', 'quartic', 'resonance_tol', 'drop_imaginary', 'results'
    }

    def __init__(self, method):
        self.base = method
        self.mol = method.mol
        self.verbose = self.mol.verbose
        self.stdout = self.mol.stdout

        self.displacement = 1e-2
        self.mass = None
        self.modes = None
        self.quartic = 'semi'
        self.resonance_tol = 1e-4
        self.drop_imaginary = True
        self.results = None

    def kernel(self):
        self.results = kernel(
            self.base,
            displacement=self.displacement,
            mass=self.mass,
            modes=self.modes,
            quartic=self.quartic,
            resonance_tol=self.resonance_tol,
            drop_imaginary=self.drop_imaginary,
            verbose=self.verbose,
        )
        return self.results
