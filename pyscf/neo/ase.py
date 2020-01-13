'''
Interface for PySCF and ASE
'''

from ase.calculators.calculator import Calculator
from pyscf.gto import Mole
from pyscf import dft

class Pyscf(Calculator):

    implemented_properties = ['energy', 'forces']
    default_parameters = {'basis': 'ccpvdz',
                          'xc': 'b3lyp'}  

    def __init__(self, **kwargs):
        Calculator.__init__(self, **kwargs)

    def calculate(self, atoms=None, properties=['energy', 'forces'],
                  system_changes=['positions', 'numbers', 'cell', 'pbc', 'charges', 'magmoms']):
        Calculator.calculate(self, atoms, properties, system_changes)
        mol = Mole()
        atoms = self.atoms.get_chemical_symbols()
        positions = self.atoms.get_positions()
        mol.atom = []
        for i in range(len(atoms)):
            mol.atom.append([atoms[i], tuple(positions[i])])
        mol.basis = self.parameters.basis
        mf = dft.RKS(mol)
        mf.xc = self.parameters.xc
        self.results['energy'] = mf.scf()
        g = mf.Gradients()
        self.results['forces'] = -g.grad()

