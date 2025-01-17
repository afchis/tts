read -p "Write docker container name: 'afchis_tts_{name}': " NAME
read -p "Write 5.z.cluster free port: " PORT
nvidia-smi
read -p "Write Nvidia-visible-devices: " NV_GPU

while true; do
    if [ -z "${NV_GPU}" ]; then
        GPUS=""
        break
    else
        GPUS="--runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=${NV_GPU}"
        break
    fi
done
while true; do
    if [ -z "${PORT}" ]; then
        PORTS=""
        break
    else
        PORTS="-p ${PORT}:22 --expose=22"
        break
    fi
done
VOLUMES="-v ${PWD}:/workspace/ -v /storage/3050/ProtopopovI/dbs/:/dbs/ -v /storage_labs/3050/ProtopopovI/runs/:/runs/"

docker run -itd --name afchis_tts_${NAME} --shm-size 32Gb \
    ${GPUS} ${VOLUMES} ${PORTS} afchis:base bash

docker cp ~/.vimrc afchis_tts_${NAME}:/root/.vimrc
docker cp ~/.tmux.conf afchis_tts_${NAME}:/root/.tmux.conf
docker cp ~/.vim/. afchis_tts_${NAME}:/root/.vim/
docker cp ~/.tmux/. afchis_tts_${NAME}:/root/.tmux/
docker cp ~/.ssh/. afchis_tts_${NAME}:/root/.ssh/
docker exec afchis_tts_${NAME} service ssh restart 

echo "Docker container name: afchis_tts_${NAME}"
echo "Connecting port: ${PORT}"


