# Use Miniconda3 as base image
FROM continuumio/miniconda3
# This line specifies the base image that will be used to build your new Docker image. 
# Here, continuumio/miniconda3 is a lightweight image that includes Miniconda3, a minimal installer for Conda.

# Set the working directory
WORKDIR /app
# This sets the working directory inside the container to /app. 
# All subsequent commands (COPY, RUN, ENTRYPOINT, etc.) will be executed in this directory.

# Copy current directory contents into the container
COPY . /app
# This copies the entire current directory (where your Dockerfile is located) into the container's /app directory.

# Create a Conda environment from pyWAXS.yml
RUN conda env create -f pyWAXS.yml
# This runs a command inside the container to create a new Conda environment based on the pyWAXS.yml file. 
# The RUN instruction will execute the command during the image build process, not when a container is started from the image.

# Activate the environment
SHELL ["conda", "run", "-n", "pyWAXS", "/bin/bash", "-c"]
# The SHELL instruction allows you to define the default shell for the subsequent RUN, CMD, and ENTRYPOINT instructions. 
# Here, it sets the shell to use the pyWAXS Conda environment.

# Expose any ports the app is listening on
EXPOSE 80
# This tells Docker that the container will listen on the specified network ports at runtime. 
# Here, it's set to port 80. You can map this port to a port on your host machine when you run the container.

# Run the script when the container launches
ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "pyWAXS", "python", "/app/apps/pyWAXSUI.py"]
# The ENTRYPOINT specifies a command that will be run when a container is started from the image. 
# Here, it's set to run your Python script pyWAXSUI.py using the pyWAXS Conda environment.
# The --no-capture-output flag is used to display the output of the conda run command in the container's stdout, making it easier to diagnose issues.