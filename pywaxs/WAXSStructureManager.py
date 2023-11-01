from pywaxs.WAXSUser import WAXSUser
import shutil, os, sys
from pathlib import Path
from ipywidgets import Dropdown
import tkinter as tk
from tkinter import filedialog

class WAXSStructureManager:
    def __init__(self, user_instance):
        if not isinstance(user_instance, WAXSUser):
            print("Error: The provided instance is not a valid WAXSUser object.")
            sys.exit(1)

        # Check for necessary attributes in the user_instance
        required_attrs = ['username', 'basePath', 'structuresPath']
        for attr in required_attrs:
            if not hasattr(user_instance, attr):
                print(f"Error: The provided WAXSUser object is missing the '{attr}' attribute.")
                sys.exit(1)

        self.username = user_instance.username
        self.basePath = user_instance.basePath
        self.structuresPath = user_instance.structuresPath
        self.structurePath = None
        self.vaspPath = None

    def addStructure(self):
        structurename = input("Enter the structure name: ")

        # Initialize Tkinter and hide the root window
        root = tk.Tk()
        root.withdraw()

        # Open file dialog to select a .vasp file
        self.vaspPath = filedialog.askopenfilename(filetypes=[("VASP files", "*.vasp")])

        # Check if a file was selected
        if not self.vaspPath:
            print("No file selected. Exiting.")
            return

        if not self.vaspPath.endswith('.vasp'):
            print("Error: The file must have a .vasp extension.")
            return

        potential_path = self.structuresPath / structurename
        if potential_path.exists():
            print(f"Structure '{structurename}' already exists. Selecting existing folder.")
        else:
            print(f"Creating new folder for structure '{structurename}'.")
            potential_path.mkdir(parents=True, exist_ok=False)
        self.structurePath = potential_path
        self.copyStructureFile()

    def copyStructureFile(self):
        if self.structurePath is None:
            print("Error: No structure has been set. Use addStructure first.")
            return

        source = Path(self.vaspPath)
        if not source.exists():
            print("Error: The source .vasp file does not exist.")
            return

        vasp_folder = self.structurePath / "vasp"
        if not vasp_folder.exists():
            vasp_folder.mkdir()

        dest = vasp_folder / source.name
        shutil.copy(source, dest)
        self.vaspPath = dest
        print(f"Copied .vasp file to {dest}")

    def replaceStructure(self, new_vaspPath, structurename):
        if not new_vaspPath.endswith('.vasp'):
            print("Error: The file must have a .vasp extension.")
            return

        new_structure_path = self.structuresPath / structurename
        if not new_structure_path.exists():
            print(f"Error: Structure '{structurename}' does not exist.")
            return

        vasp_folder = new_structure_path / "vasp"
        if not vasp_folder.exists():
            vasp_folder.mkdir()

        existing_vasp_files = list(vasp_folder.glob("*.vasp"))
        if existing_vasp_files:
            for file in existing_vasp_files:
                file.unlink()

        new_source = Path(new_vaspPath)
        if not new_source.exists():
            print("Error: The source .vasp file does not exist.")
            return

        dest = vasp_folder / new_source.name
        shutil.copy(new_source, dest)
        self.vaspPath = dest
        print(f"Replaced existing .vasp file with {new_source} at {dest}")

    def deleteStructure(self, structurename):
        structure_path = self.structuresPath / structurename
        if not structure_path.exists():
            print(f"Error: Structure '{structurename}' does not exist.")
            return

        confirmation = input(f"Are you sure? This will delete {structurename}, and all of its contents, including the simulation and .vasp files. (yes/no): ")
        if confirmation.lower() == 'yes':
            shutil.rmtree(structure_path)
            print(f"Deleted structure '{structurename}' and all its contents.")
        else:
            print("Operation cancelled.")

    def genStructureList(self):
        return [folder.name for folder in self.structuresPath.iterdir() if folder.is_dir()]

    def selectStructure(self):
        structure_list = self.genStructureList()
        if not structure_list:
            print("No structures available.")
            return
        
        dropdown = Dropdown(
            options=structure_list,
            description='Select Structure:',
        )
        display(dropdown)

        def on_change(change):
            if change['type'] == 'change' and change['name'] == 'value':
                self.structurePath = self.structuresPath / change['new']
                print(f"Structure {change['new']} selected.")
        
        dropdown.observe(on_change)

    def addStructureSim(self):
        # Placeholder for the addStructureSim method
        pass

    def deleteStructureSim(self):
        # Placeholder for the deleteStructureSim method
        pass

    def checkStructureSim(self):
        # Placeholder for the checkStructureSim method
        pass