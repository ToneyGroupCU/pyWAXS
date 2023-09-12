import requests
import json
import subprocess
import os

def get_installed_packages():
    print("Getting list of installed packages...")
    return subprocess.check_output(['pip', 'freeze']).decode().splitlines()

def parse_package_string(package_string):
    if " @ " in package_string:
        package_string = package_string.split(" @ ")[0]

    if '==' in package_string:
        package_name, version = package_string.split('==')
    else:
        package_name = package_string
        version = None
    
    return package_name, version

def check_pypi(package_name, version):
    print(f"Checking package {package_name} on PyPi...")
    url = f"https://pypi.org/pypi/{package_name}/json"
    response = requests.get(url)
    if response.status_code == 200:
        data = json.loads(response.text)
        return version is None or version in data['releases']
    return False

def check_conda(package_name, version):
    print(f"Checking package {package_name} on conda-forge...")
    url = f"https://api.anaconda.org/package/conda-forge/{package_name}"
    response = requests.get(url)
    if response.status_code == 200:
        data = json.loads(response.text)
        available_versions = [file['version'] for file in data['files']]
        return version is None or version in available_versions
    return False

def write_to_file(filename, packages):
    print(f"Writing to {filename}...")
    with open(filename, 'w') as f:
        for package in packages:
            f.write("%s\n" % package)

def categorize_packages():
    print("Categorizing packages...")
    installed_packages = get_installed_packages()

    pip_requirements = []
    conda_requirements = []
    not_found_packages = []

    for package_string in installed_packages:
        package_name, version = parse_package_string(package_string)
        if check_pypi(package_name, version):
            pip_requirements.append(package_string)
        elif check_conda(package_name, version):
            conda_requirements.append(package_string)
        else:
            not_found_packages.append(package_string)

    write_to_file('pip_requirements.txt', pip_requirements)
    write_to_file('conda_requirements.txt', conda_requirements)
    write_to_file('packageversionnotfound.txt', not_found_packages)

def export_environment_yml(yml_filename):
    print("Exporting current environment to yml file...")
    if os.path.exists(yml_filename):
        should_overwrite = input(f"{yml_filename} already exists. Overwrite? (y/n): ")
        if should_overwrite.lower() != 'y':
            return
    with open(yml_filename, 'w') as f:
        subprocess.call(['conda', 'env', 'export'], stdout=f)

def create_environment_from_yml(yml_filename, env_name):
    print(f"Creating new environment {env_name} from {yml_filename}...")
    subprocess.call(['conda', 'env', 'create', '-f', yml_filename, '-n', env_name])

def install_packages(env_name):
    print(f"Installing packages into {env_name}...")
    os.system(f'source activate {env_name} && pip install -r pip_requirements.txt')
    os.system(f'source activate {env_name} && conda install --file conda_requirements.txt')

def create_env_from_files(env_name):
    print(f"Starting process to create new environment {env_name}...")
    categorize_packages()
    export_environment_yml('base.yml')
    create_environment_from_yml('base.yml', env_name)
    install_packages(env_name)

def main(env_name):
    create_env_from_files(env_name)

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        env_name = sys.argv[1]
        main(env_name)
    else:
        print("Please provide an environment name. Usage: python envgen.py <env_name>")


'''
import requests
import json
import subprocess
import os
import argparse

def get_installed_packages():
    return subprocess.check_output(['pip', 'freeze']).decode().splitlines()

def parse_package_string(package_string):
    if " @ " in package_string:
        package_string = package_string.split(" @ ")[0]

    if '==' in package_string:
        package_name, version = package_string.split('==')
    else:
        package_name = package_string
        version = None
    
    return package_name, version

def check_pypi(package_name, version):
    url = f"https://pypi.org/pypi/{package_name}/json"
    response = requests.get(url)
    if response.status_code == 200:
        data = json.loads(response.text)
        return version is None or version in data['releases']
    return False

def check_conda(package_name, version):
    url = f"https://api.anaconda.org/package/conda-forge/{package_name}"
    response = requests.get(url)
    if response.status_code == 200:
        data = json.loads(response.text)
        available_versions = [file['version'] for file in data['files']]
        return version is None or version in available_versions
    return False

def write_to_file(filename, packages):
    with open(filename, 'w') as f:
        for package in packages:
            f.write("%s\n" % package)

def categorize_packages():
    installed_packages = get_installed_packages()

    pip_requirements = []
    conda_requirements = []
    not_found_packages = []

    for package_string in installed_packages:
        package_name, version = parse_package_string(package_string)
        if check_pypi(package_name, version):
            pip_requirements.append(package_string)
        elif check_conda(package_name, version):
            conda_requirements.append(package_string)
        else:
            not_found_packages.append(package_string)

    write_to_file('pip_requirements.txt', pip_requirements)
    write_to_file('conda_requirements.txt', conda_requirements)
    write_to_file('packageversionnotfound.txt', not_found_packages)

def export_environment_yml(yml_filename):
    if os.path.exists(yml_filename):
        should_overwrite = input(f"{yml_filename} already exists. Overwrite? (y/n): ")
        if should_overwrite.lower() != 'y':
            return
    with open(yml_filename, 'w') as f:
        subprocess.call(['conda', 'env', 'export'], stdout=f)


def create_environment_from_yml(yml_filename, env_name):
    subprocess.call(['conda', 'env', 'create', '-f', yml_filename, '-n', env_name])

def install_packages(env_name):
    os.system(f'source activate {env_name} && pip install -r pip_requirements.txt')
    os.system(f'source activate {env_name} && conda install --file conda_requirements.txt')

def create_env_from_files(env_name):
    # Create environment from base.yml
    subprocess.run(["conda", "env", "create", "-f", "base.yml", "--name", env_name])

    # Activate the new environment. Note: This may not work as expected in a Jupyter notebook
    # because each cell runs as a separate subprocess. You might need to activate the environment 
    # manually in the terminal before running cells that use this environment.
    os.system(f"conda activate {env_name}")

    # Install pip requirements
    subprocess.run(["pip", "install", "-r", "pip_requirements.txt"])
    
    # Install conda requirements
    subprocess.run(["conda", "install", "--yes", "--file", "conda_requirements.txt"])

def main():
    # create the top-level parser
    parser = argparse.ArgumentParser(description='Generate a new environment from base.yml and requirements files.')
    parser.add_argument('env_name', type=str, help='Name of the new environment to create.')

    args = parser.parse_args()
    
    categorize_packages()
    export_environment_yml('base.yml')
    create_environment_from_yml('base.yml', args.env_name)
    install_packages(args.env_name)

if __name__ == "__main__":
    main()
'''

# -- Old
'''
import requests, json, os, subprocess

pip_packages = !pip freeze
pip_requirements = []
conda_requirements = []
not_found_packages = []

for package in pip_packages:
    if " @ " in package:
        package = package.split(" @ ")[0]

    if '==' in package:
        package_name, version = package.split('==')
    else:
        package_name = package
        version = None

    # Check PyPI
    url = f"https://pypi.org/pypi/{package_name}/json"
    response = requests.get(url)
    if response.status_code == 200:
        data = json.loads(response.text)
        if version is None or version in data['releases']:
            pip_requirements.append(package)
        else:
            not_found_packages.append(package)
    else:
        # Check conda-forge
        url = f"https://api.anaconda.org/package/conda-forge/{package_name}"
        response = requests.get(url)
        if response.status_code == 200:
            data = json.loads(response.text)
            available_versions = [file['version'] for file in data['files']]
            if version is None or version in available_versions:
                conda_requirements.append(package)
            else:
                not_found_packages.append(package)
        else:
            not_found_packages.append(package)

# Write pip_requirements.txt
with open('pip_requirements.txt', 'w') as f:
    for package in pip_requirements:
        f.write("%s\n" % package)

# Write conda_requirements.txt
with open('conda_requirements.txt', 'w') as f:
    for package in conda_requirements:
        f.write("%s\n" % package)

# Write packageversionnotfound.txt
with open('packageversionnotfound.txt', 'w') as f:
    for package in not_found_packages:
        f.write("%s\n" % package)

# -------------------------------------------------------- # 
# Define the function to install using pipdd

# def install_pip(packages):
#     for package in packages:
#         try:
#             subprocess.check_call(["python", "-m", "pip", "install", package])
#             print(f'Successfully installed {package}')
#         except Exception as e:
#             print(f'Failed to install {package}: {e}')

# # Define the function to install using conda
# def install_conda(packages):
#     for package in packages:
#         try:
#             subprocess.check_call(["conda", "install", "-c", "conda-forge", package, "-y"])
#             print(f'Successfully installed {package}')
#         except Exception as e:
#             print(f'Failed to install {package}: {e}')

# # Load and install pip packages
# with open('pip_requirements.txt') as f:
#     pip_packages = [line.strip() for line in f]
#     install_pip(pip_packages)

# # Load and install conda packages
# with open('conda_requirements.txt') as f:
#     conda_packages = [line.strip() for line in f]
#     install_conda(conda_packages)

# pip_packages = !pip freeze
# with open('pip_requirements.txt', 'w') as f:
#     for package in pip_packages:
#         if " @ " in package:
#             package = package.split(" @ ")[0]
#         f.write("%s\n" % package)
# -------------------------------------------------------- # 
'''