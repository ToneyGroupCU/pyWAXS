import os, sys, re
from pathlib import Path
import importlib.util

class WAXSUser:
    def __init__(self, username=None):
        # Find the pywaxs module origin
        spec = importlib.util.find_spec("pywaxs")
        if spec and spec.origin:
            module_path = Path(spec.origin)
            rootPath = module_path.parent.parent
            if "/anaconda3/envs/" in str(rootPath):
                print("Error: The path is within the Anaconda environments directory. Exiting.")
                sys.exit(1)
            self.rootPath = rootPath / "users"
        else:
            print("Error: pywaxs module not found.")
            sys.exit(1)

        if not self.rootPath.exists():
            print("Users folder not found in the expected path. Creating a new 'users' folder.")
            self.rootPath.mkdir(parents=True, exist_ok=False)

        # rest of the code
        self.username = None
        self.basePath = None
        self.projectsPath = None
        self.structuresPath = None
        self.notebooksPath = None
        self.existing_users = []  # Placeholder for the method to get existing users
        self.subdirs = ['projects', 'notebooks', 'structures']
        
        if username:
            self.set_username(username)  # Placeholder for the method to set username
        else:
            self.prompt_username()  # Placeholder for the method to prompt username
    
    def get_existing_users(self):
        """Get the list of existing usernames in the 'users' directory."""
        return [user.name for user in self.rootPath.iterdir() if user.is_dir()]
    
    def set_username(self, username):
        """Set the username if it is valid, otherwise prompt the user to re-enter."""
        if self.is_valid_username(username):
            case_sensitive_match = [u for u in self.existing_users if u == username]
            case_insensitive_match = [u for u in self.existing_users if u.lower() == username.lower()]
            
            if case_sensitive_match:
                self.username = username
                self.basePath = self.rootPath / username
                self.check_subdirs()
                self.set_subdir_paths()
            elif case_insensitive_match:
                choice = input(f"'{case_insensitive_match[0]}' exists. Create a new username with name '{username}' or load existing '{case_insensitive_match[0]}'? ")
                if choice.lower() == 'load existing':
                    self.username = case_insensitive_match[0]
                    self.basePath = self.rootPath / self.username
                    self.check_subdirs()
                    self.set_subdir_paths()
                else:
                    self.create_new_user(username)
            else:
                self.create_new_user(username)
        else:
            print("Invalid username. Please enter a valid username.")
    
    def is_valid_username(self, username):
        """Check if the username is valid."""
        pattern = re.compile("^[a-zA-Z][a-zA-Z0-9_-]{0,29}$")
        return bool(pattern.match(username))
    
    def create_new_user(self, username):
        """Create a new user and their subdirectories."""
        self.username = username
        self.basePath = self.rootPath / username
        self.basePath.mkdir(parents=True, exist_ok=True)
        self.existing_users.append(username)
        for subdir in self.subdirs:
            (self.basePath / subdir).mkdir(parents=True, exist_ok=True)
        self.set_subdir_paths()
    
    def prompt_username(self):
        """Prompt the user to enter a username."""
        while not self.username:
            username = input("Please enter a username: ")
            self.set_username(username)
    
    def check_subdirs(self):
        """Check if the subdirectories exist for an existing user."""
        for subdir in self.subdirs:
            sub_path = self.basePath / subdir
            if not sub_path.exists():
                print(f"Directory '{subdir}' does not exist. Would you like to create it? (yes/no)")
                choice = input()
                if choice.lower() == 'yes':
                    sub_path.mkdir(parents=True, exist_ok=True)

    def set_subdir_paths(self):
        """Set the pathlib variable attributes for the subdirectories."""
        self.projectsPath = self.basePath / 'projects'
        self.notebooksPath = self.basePath / 'notebooks'
        self.structuresPath = self.basePath / 'structures'

# Creating the WAXSProject subclass that inherits from WAXSUser
class WAXSProject(WAXSUser):
    def __init__(self, user_instance, projectname=None):
        # Inherit attributes from WAXSUser
        super().__init__(user_instance.username)
        
        self.projectname = None
        self.projectPath = None
        self.dataPath = None
        self.maskPath = None
        self.poniPath = None
        self.analysisPath = None
        self.reducedPath = None
        self.integratedPath = None
        self.imagesPath = None
        self.peakposPath = None
        self.peakfitPath = None
        self.phasesPath = None
        self.topasPath = None
        self.subdirs = ['data', 'mask', 'poni', 'analysis']
        self.analysis_subdirs = ['reduced', 'integrated', 'images', 'peakpos', 'peakfit', 'phases', 'topas']
        
        if projectname:
            self.set_projectname(projectname)
        else:
            self.prompt_projectname()
    
    def set_projectname(self, projectname):
        """Set the project name and manage its subdirectories."""
        project_path_candidate = self.projectsPath / projectname
        if project_path_candidate.exists():
            self.projectname = projectname
            self.projectPath = project_path_candidate
            self.set_project_subdir_paths()
        else:
            self.create_new_project(projectname)
    
    def create_new_project(self, projectname):
        """Create a new project and its subdirectories."""
        self.projectname = projectname
        self.projectPath = self.projectsPath / projectname
        self.projectPath.mkdir(parents=True, exist_ok=True)
        
        for subdir in self.subdirs:
            (self.projectPath / subdir).mkdir(parents=True, exist_ok=True)
        
        analysis_path = self.projectPath / 'analysis'
        for analysis_subdir in self.analysis_subdirs:
            (analysis_path / analysis_subdir).mkdir(parents=True, exist_ok=True)
        
        self.set_project_subdir_paths()
    
    def set_project_subdir_paths(self):
        self.dataPath = self.projectPath / 'data'
        self.maskPath = self.projectPath / 'mask'
        self.poniPath = self.projectPath / 'poni'
        self.analysisPath = self.projectPath / 'analysis'
        self.reducedPath = self.analysisPath / 'reduced'
        self.integratedPath = self.analysisPath / 'integrated'
        self.imagesPath = self.analysisPath / 'images'
        self.peakposPath = self.analysisPath / 'peakpos'
        self.peakfitPath = self.analysisPath / 'peakfit'
        self.phasesPath = self.analysisPath / 'phases'
        self.topasPath = self.analysisPath / 'topas'
    
    def prompt_projectname(self):
        """Prompt the user to enter a project name."""
        while not self.projectname:
            projectname = input("Please enter a project name: ")
            self.set_projectname(projectname)