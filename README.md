# Desde la carpeta del proyecto (donde está el Dockerfile)
docker build -t lisa_task1:0.1 .

############################

docker compose build --no-cache run
docker compose up --build run

up crea el contenedor run y ejecuta automáticamente el ENTRYPOINT de tu imagen (tu entrypoint.sh).
--build recompila si cambiaste algo del Dockerfile.

# Abrir un bash dentro del contenedor (sin ejecutar tu entrypoint)
docker run -it --entrypoint bash lisa_task1:0.1

############################

# etiqueta para Synapse (nombre sugerido: task1)
docker tag lisa_task1:0.1 docker.synapse.org/syn68765796/task1:0.1


# login y push
docker login docker.synapse.org
docker push docker.synapse.org/syn68765796/task1:0.1
docker push docker.synapse.org/syn68765796/task1:latest


########### FINAL ###################
docker build -t docker.synapse.org/syn68765796/conda20:0.1 .
docker run -it --rm --gpus all -v E:\Datathon\LISA\input\task-1-val:/input:ro -v E:\Datathon\LISA\output:/output:rw docker.synapse.org/syn68765796/conda20:0.1


docker run -it -v E:\Datathon\LISA\input\task-1-val:/input:ro -v E:\Datathon\LISA\output:/output:rw docker.synapse.org/syn68765796/conda20:0.1
docker run --rm -v /mnt/e/Datathon/LISA/input/task-1-val:/input:ro -v /mnt/e/Datathon/LISA/output:/output:rw docker.synapse.org/syn68765796/conda20:0.1
docker run -it -v /mnt/e/Datathon/LISA/input/task-1-val:/input:ro -v /mnt/e/Datathon/LISA/output:/output:rw docker.synapse.org/syn68765796/conda20:0.1


docker push docker.synapse.org/syn68765796/conda20:0.1



# 1) Construir imagen local para debug (con tu código montado):
docker compose -f docker-compose.yml up --build dev

# 2) Probar ejecución "tipo Synapse" contra tu dataset local:
docker compose -f docker-compose.yml up run

# 3) abrir el terminal
docker compose -f docker-compose.yml run --rm run bash
