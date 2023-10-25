import os
from pathlib import Path
from typing import Optional, Tuple, List
import pandas as pd
import math
import numpy as np

class WAXSFileManager:
    @staticmethod
    def create_folder_structure(basePath: Path, folder_name: str) -> None:
        main_folder = os.path.join(basePath, folder_name)
        
        if not os.path.exists(main_folder):
            os.makedirs(main_folder)
        
        subfolders = ['poni', 'mask', 'data', 'analysis', 'poscar', 'stitch']
        subsubfolders = ['hdf5', 'png', 'simulation']

        for sub in subfolders:
            os.makedirs(os.path.join(main_folder, sub), exist_ok=True)
            if sub == 'analysis':
                analysisPath = Path(main_folder).joinpath('analysis')
                for subsub in subsubfolders:
                    os.makedirs(os.path.join(analysisPath, subsub), exist_ok=True)
    
    @staticmethod
    def load_int_file(filePath: Path, wavelength: float) -> pd.DataFrame:
        df = pd.read_csv(filePath, skiprows=2, header=None, delimiter='\s+')
        df.columns = ['twotheta', 'intensity', 'error']
        df['q'] = (4 * math.pi / wavelength) * np.sin(np.radians(df['twotheta'] / 2))
        df.attrs['wavelength'] = wavelength
        df.attrs['filePath'] = str(filePath)
        
        return df
    
    @staticmethod
    def update_filePath(folderPath: Path, extension: str, scanID: Optional[str] = None) -> Optional[Path]:
        files = list(folderPath.glob(f'*.{extension}'))

        if scanID:
            files = [f for f in files if f"_{scanID}_" in f.name]

        if len(files) > 1:
            raise ValueError(f"Multiple .{extension} files found in the folder.")
        elif len(files) == 0:
            raise ValueError(f"No .{extension} files found in the folder.")
        
        return folderPath.joinpath(files[0].name)
    
    @staticmethod
    def generate_projectPaths(basePath: Path, project_name: str) -> Tuple[Path, List[Path]]:
        WAXSFileManager.create_folder_structure(basePath, project_name)
        projectPath = basePath.joinpath(project_name)

        # Define various project paths
        dataPath = projectPath.joinpath('data')
        poniPath = projectPath.joinpath('poni')
        maskPath = projectPath.joinpath('mask')
        analysisPath = projectPath.joinpath('analysis')
        poscarPath = projectPath.joinpath('poscar')

        hdf5Path = analysisPath.joinpath('hdf5')
        pngPath = analysisPath.joinpath('png')
        simulationPath = analysisPath.joinpath('simulation')

        # Update paths based on existing files
        try:
            poniPath = WAXSFileManager.update_filePath(poniPath, 'poni')
            maskPath = WAXSFileManager.update_filePath(maskPath, 'edf')
            dataPath = WAXSFileManager.update_filePath(dataPath, 'tiff')

            print(f"Updated dataPath: {dataPath}")
            print(f"Updated poniPath: {poniPath}")
            print(f"Updated maskPath: {maskPath}")
            
        except ValueError as e:
            print(e)

        PathList = [projectPath, dataPath, poniPath, maskPath, analysisPath, poscarPath, hdf5Path, pngPath, simulationPath]
        return projectPath, PathList
