## -- Module WAXS Classes -- ##
from WAXSSearch import WAXSSearch
from WAXSDiffSim import WAXSDiffSim

# - Import Relevant Modules"
import xarray as xr
import numpy as np
import pathlib, os
from pathlib import Path
from typing import Optional
import matplotlib.pyplot as plt
from math import cos, sin, radians, degrees
import pandas as pd

# - pymatgen: https://pymatgen.org/
# from pymatgen import Structure, Lattice
from pymatgen.core import Structure, Lattice, Molecule, IMolecule
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.io.cif import CifWriter

# Full definition of MoleculeConstructor with new methods for molecule construction
class MoleculeConstructor:
    def __init__(self, filepath, use_experimental=True):
        """
        Initialize the MoleculeConstructor class.
        
        Args:
        - filepath (str or pathlib.Path): The path to the Excel file containing the bond data.
        """
        self.filepath = filepath if isinstance(filepath, pathlib.Path) else pathlib.Path(filepath)
        self.df_bond_distances = None
        self.df_bond_angles = None
        # self.atom_df = pd.DataFrame(columns=['atom_name', 'element', 'x_frac', 'y_frac', 'z_frac', 
        #                                      'coordinated_atoms', 'bond_distance_ref', 'bond_angle', 'source'])
        self.atom_df = pd.DataFrame([], columns=['atom_name', 
                                                 'element',
                                                 'x_frac', 
                                                 'y_frac', 
                                                 'z_frac', 
                                                 'coordinated_atoms', 
                                                 'bond_distance_ref', 
                                                 'bond_angle', 
                                                 'source'])
        self.use_experimental = use_experimental
        self.visited_atoms = set()
        self.load_molecule_data(self.filepath)
        
    def load_molecule_data(self, filepath):
        """
        Load bond distances and bond angles from an Excel file into Pandas dataframes.
        
        Args:
        - filepath (str or pathlib.Path): The path to the Excel file containing the bond data.
        """
        # Handle both string and pathlib.Path types for filepath
        filepath = str(filepath) if isinstance(filepath, pathlib.Path) else filepath
        
        # Read the bond distances sheet
        df_bond_distances = pd.read_excel(filepath, sheet_name='bond_distances', index_col=[0, 1])
        df_bond_distances.columns = ['distance_angstr_experimental', 'distance_angstr_ideal']
        
        # Read the bond angles sheet
        df_bond_angles = pd.read_excel(filepath, sheet_name='bond_angles', index_col=[0, 1, 2])
        df_bond_angles.columns = ['angle_deg_experimental', 'angle_deg_ideal']
        
        self.df_bond_angles = df_bond_angles
        self.df_bond_distances = df_bond_distances

    def set_origin(self, origin_atom, element):
        """
        Set the origin atom at the center of the unit cell in fractional coordinates.
        """
        new_row = pd.DataFrame([{
            'atom_name': origin_atom,
            'element': element,
            'x_frac': 0.5,
            'y_frac': 0.5,
            'z_frac': 0.5,
            'coordinated_atoms': [],
            'bond_distance_ref': None,
            'bond_angle': None,
            'source': None
        }])
        self.atom_df = pd.concat([self.atom_df, new_row]).reset_index(drop=True)

    def add_atom(self, current_atom, new_atom, element, angle=None, ref_atom=None):
        """
        Add a new atom based on the bond length and angle from the current atom.
        """
        # Get the distance from the data
        distance = self.get_distance(current_atom, new_atom)
        
        # Placeholder logic to calculate new coordinates (to be refined later)
        current_coords = self.atom_df[self.atom_df['atom_name'] == current_atom][['x_frac', 'y_frac', 'z_frac']].values[0]
        new_coords = current_coords + [distance, 0, 0]
        
        # Create a new DataFrame row
        new_row = pd.DataFrame([{
            'atom_name': new_atom,
            'element': element,
            'x_frac': new_coords[0],
            'y_frac': new_coords[1],
            'z_frac': new_coords[2],
            'coordinated_atoms': [current_atom],
            'bond_distance_ref': current_atom,
            'bond_angle': angle,
            'source': 'Experimental' if self.use_experimental else 'Ideal'
        }])
        
        # Concatenate the new row with the existing DataFrame
        self.atom_df = pd.concat([self.atom_df, new_row]).reset_index(drop=True)
        
        # Update the coordinated_atoms list for the current_atom
        self.atom_df.loc[self.atom_df['atom_name'] == current_atom, 'coordinated_atoms'].apply(lambda x: x.append(new_atom))
        
    def get_distance(self, atom1, atom2):
        """
        Retrieve the bond distance between atom1 and atom2 from the DataFrame.
        """
        source_col = 'distance_angstr_experimental' if self.use_experimental else 'distance_angstr_ideal'
        distance = self.df_bond_distances.loc[(atom1, atom2)][source_col] if (atom1, atom2) in self.df_bond_distances.index else \
                   self.df_bond_distances.loc[(atom2, atom1)][source_col] if (atom2, atom1) in self.df_bond_distances.index else \
                   None
        return distance

    def get_angle(self, atom1, atom2, atom3):
        """
        Retrieve the bond angle between atom1, atom2, and atom3 from the DataFrame.
        """
        source_col = 'angle_deg_experimental' if self.use_experimental else 'angle_deg_ideal'
        angle = self.df_bond_angles.loc[(atom1, atom2, atom3)][source_col] if (atom1, atom2, atom3) in self.df_bond_angles.index else None
        return angle

    '''
    def construct_molecule(self, current_atom, element):
        """
        Recursive function to construct the molecule starting from the current atom.
        """
        # Mark the current atom as visited
        self.visited_atoms.add(current_atom)
        
        # Identify all valid adjacent atoms based on rules
        adjacent_atoms_i_j = self.df_bond_angles.reset_index().loc[(self.df_bond_angles.reset_index()['atom_i'] == current_atom)]['atom_j'].tolist()
        adjacent_atoms_j_k = self.df_bond_angles.reset_index().loc[(self.df_bond_angles.reset_index()['atom_j'] == current_atom)]['atom_k'].tolist()
        
        # Combine and remove duplicates and visited atoms
        adjacent_atoms = list(set(adjacent_atoms_i_j + adjacent_atoms_j_k) - self.visited_atoms)
        
        # Sort atoms by their frequency in df_bond_angles
        atom_freq = self.df_bond_angles.reset_index()['atom_i'].value_counts().to_dict()
        adjacent_atoms = sorted(adjacent_atoms, key=lambda x: atom_freq.get(x, 0))
        
        # Loop through each adjacent atom to calculate its coordinates
        for new_atom in adjacent_atoms:
            # Get distance and angle
            distance = self.get_distance(current_atom, new_atom)
            angle = self.get_angle(current_atom, new_atom, element)  # Assuming element as the ref_atom for now
            
            # Placeholder logic to calculate new coordinates
            current_coords = self.atom_df[self.atom_df['atom_name'] == current_atom][['x_frac', 'y_frac', 'z_frac']].values[0]
            new_coords = current_coords + [distance, 0, 0]  # Placeholder
            
            # Add the new atom to the DataFrame
            self.add_atom(current_atom, new_atom, element, angle, current_atom)
            
            # Recursive call
            self.construct_molecule(new_atom, element)
    '''

    def construct_molecule(self, current_atom, element):
        self.visited_atoms.add(current_atom)
        adjacent_atoms_i_j = self.df_bond_angles.reset_index().loc[(self.df_bond_angles.reset_index()['atom_i'] == current_atom)]['atom_j'].tolist()
        adjacent_atoms_j_k = self.df_bond_angles.reset_index().loc[(self.df_bond_angles.reset_index()['atom_j'] == current_atom)]['atom_k'].tolist()
        adjacent_atoms = list(set(adjacent_atoms_i_j + adjacent_atoms_j_k) - self.visited_atoms)
        atom_freq = self.df_bond_angles.reset_index()['atom_i'].value_counts().to_dict()
        adjacent_atoms = sorted(adjacent_atoms, key=lambda x: atom_freq.get(x, 0))
        for new_atom in adjacent_atoms:
            distance = self.get_distance(current_atom, new_atom)
            angle = self.get_angle(current_atom, new_atom, element)  
            self.add_atom(current_atom, new_atom, element, angle, current_atom)
            self.construct_molecule(new_atom, element)

    def save_to_cif(self, cif_directory, cif_name, lattice_parameter=20.0):
        """
        Save the constructed molecule to a CIF file.
        
        Args:
        - cif_directory (str or pathlib.Path): The directory where the CIF file will be saved.
        - cif_name (str): The name of the CIF file.
        - lattice_parameter (float): The lattice parameter for the cubic cell in Angstroms.
        """
        # Handle both string and pathlib.Path types for cif_directory
        cif_directory = str(cif_directory) if isinstance(cif_directory, pathlib.Path) else cif_directory
        
        # Create full path for the CIF file
        cif_filepath = f"{cif_directory}/{cif_name}.cif"
        
        # Extract species and coordinates from the DataFrame
        species = self.atom_df['element'].tolist()
        coords = self.atom_df[['x_frac', 'y_frac', 'z_frac']].values.tolist()
        
        # Create a Molecule object
        molecule = Molecule(species, coords)
        
        # Create a Structure object from the Molecule object
        lattice = Lattice.cubic(lattice_parameter)  # Use the provided lattice parameter
        structure = Structure(lattice, species, coords)
        
        # Convert the structure to a CIF file
        writer = CifWriter(structure)
        writer.write_file(cif_filepath)