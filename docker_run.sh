#!/bin/bash
docker run -ti --gpus '"device='$1'"' -v /sac:/app --ipc=host --name $2 sac /bin/bash
# ex) docker run -ti --gpus '"device='$1'"' -v <put_your_working_directory>:/app --ipc=host --name $2 <put_your_docker_image_name> /bin/bash