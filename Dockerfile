FROM --platform=linux/amd64 mambaorg/micromamba:1.5.8-focal-cuda-12.1.1

USER root

# Clean up
RUN rm -f /etc/apt/sources.list.d/*.list

# Utilities
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential sudo curl git htop less rsync screen vim nano wget

# Environment
COPY environment.yaml /tmp/environment.yaml
RUN micromamba install -y -n base -f /tmp/environment.yaml -v && micromamba clean -qya

# Google Cloud SDK
RUN curl https://dl.google.com/dl/cloudsdk/channels/rapid/downloads/google-cloud-cli-418.0.0-linux-x86_64.tar.gz > /tmp/google-cloud-sdk.tar.gz
RUN mkdir -p /usr/local/gcloud \
  && tar -C /usr/local/gcloud -xvf /tmp/google-cloud-sdk.tar.gz \
  && /usr/local/gcloud/google-cloud-sdk/install.sh
ENV PATH $PATH:/usr/local/gcloud/google-cloud-sdk/bin

# Workdir
RUN mkdir /srv/repo/ && chmod 777 /srv/repo
ENV PYTHONPATH $PYTHONPATH:/srv/repo
ENV PATH $MAMBA_ROOT_PREFIX/bin:$PATH
WORKDIR /srv/repo
