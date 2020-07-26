ARG BASE_IMAGE
ARG HTTP_PROXY
ARG HTTPS_PROXY
ARG NO_PROXY
FROM $BASE_IMAGE

# Using the tensorflow base image, the current directory is /tf.
# That is where Jupyter will open.
# Jupyter will display it as simply / ,
# but checking the current working directory in code will confirm it is /tf.

ENV http_proxy=$HTTP_PROXY
ENV https_proxy=$HTTPS_PROXY
ENV no_proxy=$NO_PROXY

RUN pip uninstall --yes enum34
RUN cat /etc/apt/sources.list \
    && ls /etc/apt/sources.list.d \
    && add-apt-repository --remove 'https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64  Release' \
    && add-apt-repository --remove 'https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64  Release' \
    && add-apt-repository --remove cuda.list \
    && add-apt-repository --remove nvidia-ml.list \
    && rm /etc/apt/sources.list.d/nvidia-ml.list \
    && rm /etc/apt/sources.list.d/cuda.list \
    # && add-apt-repository "ppa:jonathonf/ffmpeg-4" \
    && apt-get update && apt-get install --assume-yes ffmpeg
RUN (python -c "import sphinx" && pip install --upgrade sphinx) || echo "Sphinx is not installed."

RUN mkdir ./video_files
RUN touch ./video_files/ifyouareseeingthisdirectoryisnotmounted
# If we later --mount type=bind to map a host directory to /tf/video_files,
# then dummyfile should NOT appear, as the /tf/video_files on the image is shadowed by the bind mount.
# https://docs.docker.com/storage/bind-mounts/#mount-into-a-non-empty-directory-on-the-container
VOLUME /video_files
# If any build steps change the data within the volume after it has been declared, those changes will be discarded.

# .dockerignore applies to COPY
COPY ./ ./prednet
COPY ./src/prednet/tests/resources/ ./test_videos
RUN pip install --no-cache ./prednet
