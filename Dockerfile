# Use Miniconda3 as base image
FROM continuumio/miniconda3

# Set the working directory
WORKDIR /app

# Copy current directory contents into the container
COPY . /app

# Create a Conda environment from pyWAXS.yml
RUN conda env create -f pyWAXS.yml

# Activate the environment
SHELL ["conda", "run", "-n", "pyWAXS", "/bin/bash", "-c"]

# Expose any ports the app is listening on
EXPOSE 80

# Run the script when the container launches
ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "pyWAXS", "python", "/app/apps/pyWAXSUI.py"]