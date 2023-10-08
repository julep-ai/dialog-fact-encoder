##########
## BASE ##
##########

FROM nvidia/cuda:12.2.0-devel-ubuntu22.04 

ENV PYTHONUNBUFFERED=1

# SYSTEM
RUN apt-get update --yes --quiet && DEBIAN_FRONTEND=noninteractive apt-get install --yes --quiet --no-install-recommends \
    software-properties-common \
    build-essential ca-certificates \
 && rm -rf /var/lib/apt/lists/*

# PYTHON 3.10
RUN add-apt-repository --yes ppa:deadsnakes/ppa && apt-get update --yes --quiet
RUN DEBIAN_FRONTEND=noninteractive apt-get install --yes --quiet --no-install-recommends \
    python3.10 \
    python3.10-dev \
    python3.10-distutils \
    python3.10-lib2to3 \
    python3.10-gdbm \
    python3.10-tk \
    pip

############
## SERVER ##
############

ENV APP_HOME /app
WORKDIR $APP_HOME
COPY ./requirements* $APP_HOME/

RUN pip install -r requirements.txt 
RUN pip install -r requirements-dev.txt
RUN rm -rf /root/.cache/pip

COPY ./embedder $APP_HOME/embedder
COPY ./training $APP_HOME/training

# RUN umount /usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1 && \
#     rm -rf /usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1

CMD [ \
    "jupyter", "lab", \
    "--ip", "0.0.0.0", "--port", "8888", \
    "--allow-root", "--no-browser", "--NotebookApp.token=''", "--NotebookApp.password=''" \
]
