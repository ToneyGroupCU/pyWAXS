import sys
import os

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Get the parent directory
parent_dir = os.path.dirname(script_dir)

# Construct the path to the /main/ directory
main_dir = os.path.join(parent_dir, 'main')

# Add the /main/ directory to sys.path
sys.path.append(main_dir)

# Test the imports
try:
    import WAXSReduce # type: ignore
    from WAXSReduce import Integration1D # type: ignore
    import WAXSDiffSim # type: ignore
    import WAXSAFF # type: ignore
    # from pyWAXSimUI import SimWindow
    # from pyWAXSUI import MyCanvas

    print("All modules imported successfully.")
except ImportError as e:
    print(f"An error occurred: {e}")
