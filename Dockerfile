# ==== Base: PyTorch + CUDA (GPU) ====
FROM pytorch/pytorch:2.2.1-cuda12.1-cudnn8-runtime

# Evita interacciones al instalar paquetes
ARG DEBIAN_FRONTEND=noninteractive

# Zona horaria configurable
ARG CONTAINER_TIMEZONE=Etc/UTC
ENV TZ=${CONTAINER_TIMEZONE}

# Librerías del sistema y utilidades (OpenCV headless, JPEG/PNG, GL)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        tzdata ffmpeg libsm6 libxext6 libxrender-dev \
        libgl1-mesa-glx libglib2.0-0 libgtk2.0-dev \
        libjpeg-dev zlib1g-dev libpng-dev \
        git curl wget unzip ca-certificates && \
    ln -snf /usr/share/zoneinfo/${TZ} /etc/localtime && echo ${TZ} > /etc/timezone && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Python deps
COPY ./requirements.txt /tmp/requirements.txt
RUN python -m pip install --upgrade pip && \
    # fija versiones críticas que ya probaste
    pip install --no-deps monai==1.5.0 numpy==1.26.4 && \
    pip install -r /tmp/requirements.txt

# todo vive aquí
WORKDIR /my_solution

# Código
COPY ./src ./src
COPY ./2.csv_creation.py ./2.csv_creation.py
COPY ./3.pipeline3dto2d.py ./3.pipeline3dto2d.py
COPY ./train.py ./train.py

# Weights
RUN mkdir -p /my_solution/models/new_model_bbox_giou_brain0.1_l0.1
COPY ./results/new_model_bbox_giou_brain0.1_l0.1 /my_solution/models/new_model_bbox_giou_brain0.1_l0.1

# Entrypoint + predictor dentro de /my_solution
COPY ./entrypoint.sh /my_solution/entrypoint.sh
COPY ./predict.py /my_solution/predict.py
RUN chmod +x /my_solution/entrypoint.sh

ENV INPUT_DIR=/input \
    OUTPUT_DIR=/output \
    MODEL_DIR=/my_solution/models/new_model_bbox_giou_brain0.1_l0.1


ENTRYPOINT ["/my_solution/entrypoint.sh"]
