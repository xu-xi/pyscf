'''
Interface for PySCF and ASE
'''

from ase.calculators.calculator import Calculator
from ase.units import Bohr, Hartree
from pyscf import neo
from pyscf import gto, dft

class Pyscf_NEO(Calculator):

    implemented_properties = ['energy', 'forces']
    default_parameters = {'basis': 'ccpvdz',
                          'charge': 0,
                          'spin': 0,
                          'xc': 'b3lyp',
                          'quantum_nuc': 'all'}  


    def __init__(self, **kwargs):
        Calculator.__init__(self, **kwargs)

    def calculate(self, atoms=None, properties=['energy', 'forces'],
                  system_changes=['positions', 'numbers', 'cell', 'pbc', 'charges', 'magmoms']):
        Calculator.calculate(self, atoms, properties, system_changes)
        mol = neo.Mole()
        atoms = self.atoms.get_chemical_symbols()
        positions = self.atoms.get_positions()
        mol.atom = []
        for i in range(len(atoms)):
            if atoms[i] == 'Mu':
                mol.atom.append(['H@0', tuple(positions[i])])
            elif atoms[i] == 'D':
                mol.atom.append(['H@2', tuple(positions[i])])
            else:
                mol.atom.append(['%s%i' %(atoms[i],i), tuple(positions[i])])
        mol.basis = self.parameters.basis
        mol.build(quantum_nuc = self.parameters.quantum_nuc, charge = self.parameters.charge, spin = self.parameters.spin)
        if self.parameters.spin == 0:
            mf = neo.CDFT(mol)
        else:
            mf = neo.CDFT(mol, restrict = False)
        mf.xc = self.parameters.xc
        self.results['energy'] = mf.scf()*Hartree
        g = mf.Gradients()
        self.results['forces'] = -g.grad()*Hartree/Bohr

class Pyscf_DFT(Calculator):

    implemented_properties = ['energy', 'forces']
    default_parameters = {'mf':'RKS',
                          'basis': 'ccpvdz',
                          'charge': 0,
                          'spin': 0,
                          'xc': 'b3lyp',
                         }  


    def __init__(self, **kwargs):
        Calculator.__init__(self, **kwargs)

    def calculate(self, atoms=None, properties=['energy', 'forces'],
                  system_changes=['positions', 'numbers', 'cell', 'pbc', 'charges', 'magmoms']):
        Calculator.calculate(self, atoms, properties, system_changes)
        mol = gto.Mole()
        atoms = self.atoms.get_chemical_symbols()
        positions = self.atoms.get_positions()
        mol.atom = []
        for i in range(len(atoms)):
            if atoms[i] == 'D':
                mol.atom.append(['H@2', tuple(positions[i])])
            else:
                mol.atom.append(['%s' %(atoms[i]), tuple(positions[i])])
        mol.basis = self.parameters.basis
        mol.build(charge = self.parameters.charge, spin = self.parameters.spin)
        if self.parameters.spin != 0:
            mf = dft.UKS(mol)
        else:
            mf = dft.RKS(mol)
        mf.xc = self.parameters.xc
        self.results['energy'] = mf.scf()*Hartree
        g = mf.Gradients()
        self.results['forces'] = -g.grad()*Hartree/Bohr

