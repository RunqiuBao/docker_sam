# allow GUI application in docker container to reach host device.
# note GUI might still fail, try launching a few more times.
xhost +local:docker

docker run -it \
    --env="DISPLAY=$DISPLAY" \
    --env="QT_X11_NO_MITSHM=1" \
    --gpus all \
    --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
    --volume="/usr/local:/usr/local" \
    --volume="$(pwd)/../SegmentAnythingPortable:/root/code/SegmentAnythingPortable:rw" \
    --volume="/home/mujin/tmptmp/finetuneSam:/root/data2:rw" \
    --volume="$(pwd)/../data:/root/data:rw" \
    --name=sampytorch \
    sampytorch \
    bash
echo "done"
