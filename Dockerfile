# Download from docker hub an image with Ubuntu 20.04 and python 3.7
#Base image for running models (not for compiling them)

# FROM  bsl546/docker_energym_base:v02

ARG BASE_IMAGE
ARG INSTALLER_IMAGE
ARG VALIDATOR_IMAGE

FROM $BASE_IMAGE as base
FROM $INSTALLER_IMAGE as installer
FROM $VALIDATOR_IMAGE as validator

FROM base

# install software needed for the workload
# this example is installing figlet
RUN apt-get update && \
    apt-get install --no-install-recommends --no-install-suggests -yq \
        figlet && \
    rm -rf /var/lib/apt/lists/* && \
    apt-get purge --auto-remove && \
    apt-get clean && \
    figlet Singularity

# get the installation scripts
COPY --from=installer /installer /opt/microsoft/_singularity/installations/

# install components required for running this image in Singularity
RUN /opt/microsoft/_singularity/installations/singularity/installer.sh

# get the validation scripts
COPY --from=validator /validations /opt/microsoft/_singularity/validations/

# optionally set validation arguments to run additional checks for Nvidia drivers
ENV SINGULARITY_IMAGE_ACCELERATOR="NVIDIA"

# run the validation
RUN /opt/microsoft/_singularity/validations/validator.sh

WORKDIR /home/lesong/energym
ADD . /home/lesong/energym

RUN python setup.py install

CMD [ "/bin/bash" ]
