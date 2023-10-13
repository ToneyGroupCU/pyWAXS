#!/bin/bash # This is a bash script
#SBATCH --job-name=python_job # Give our job a name in the scheduler
#SBATCH --nodes=1 # Designate how many nodes we wanted
#SBATCH --ntasks-per-node=1 # Estimated tasks per node.
#SBATCH --cpus-per-task=1 # Number of CPUS per task that will be required
#SBATCH --time=01:00:00 # time format --time=dd-hh:mm:ss (all optional except for seconds)
#SBATCH --mem=1000
#SBATCH --partition=general
#SBATCH --output=python_job.out # Output file name (standard output from any command)
#SBATCH --error=python_job.err # Error file name (standard error from any command)
##SBATCH --mail-type=ALL
##SBATCH --mail-user=keith.white@colorado.edu

# Assuming Conda is installed and in your path
# Create the environment from the .yml file
conda env create -f pyWAXS.yml

# Activate the environment
source activate pyWAXS

# Change to the directory where the .py script is
cd $DIR

# Run the Python script
python3 script.py

# :bash command: 
# sbatch my_script.sh