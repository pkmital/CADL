FROM continuumio/anaconda3

RUN apt-get update && apt-get install -y \
        pkg-config \
        libfreetype6-dev \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

ENV TENSORFLOW_VERSION 1.1.0
RUN pip install tensorflow==$TENSORFLOW_VERSION 
# RUN conda update conda; conda update --all

COPY jupyter_notebook_config.py /root/.jupyter/

# Jupyter has issues with being run directly:
# https://github.com/ipython/ipython/issues/7062
# We just add a little wrapper script.
COPY run_jupyter.sh /

# tensorboard
EXPOSE 6006

# jupyter
EXPOSE 8888
EXPOSE 8889

WORKDIR "/notebooks"

CMD ["/bin/bash"]
