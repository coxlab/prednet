ARG BASE_IMAGE=dahanna/python-alpine-package:alpine-python3-dev-git
# This Dockerfile has a default BASE_IMAGE, but you can override it with --build-arg.

# To be used in the FROM, naturally an ARG must come before the FROM.
# However, any *other* ARG above the FROM will be silently ignored.
# So it's very important that the ARGs are below the FROM.
# https://docs.docker.com/engine/reference/builder/#understand-how-arg-and-from-interact
FROM $BASE_IMAGE
ARG ETC_ENVIRONMENT_LOCATION

# Using the tensorflow base image, the current directory is /tf.
# That is where Jupyter will open.
# Jupyter will display it as simply / ,
# but checking the current working directory in code will confirm it is /tf.

COPY dockerfiles/before_script.sh .

# .dockerignore keeps .tox and so forth out of the COPY.
COPY . prednet
# If we ran before_script in a separate RUN before the COPY of the code,
# then that layer could stay cached when the repo contents changed,
# but it's more valuable to keep all the environment variables confined to a single RUN.
# before_script.sh shouldn't take long to run anyway.

# The before_script.sh script sets several environment variables.
# Environment variables do *not* persist across Docker RUN lines.
# See also https://vsupalov.com/set-dynamic-environment-variable-during-docker-image-build/
RUN if [ -z ${FTP_PROXY+ABC} ]; then echo "FTP_PROXY is unset, so not doing any shenanigans."; else SETTER="SSH_PRIVATE_DEPLOY_KEY=${FTP_PROXY}"; fi \
    && ${SETTER} . ./before_script.sh \
    && pip install --no-cache-dir ./prednet \
    && (ssh-add -D || echo "ssh-add -D failed, hopefully because we never installed openssh-client in the first place.")

RUN mkdir ./video_files \
    && touch ./video_files/ifyouareseeingthisdirectoryisnotmounted
# If we later --mount type=bind to map a host directory to /tf/video_files,
# then dummyfile should NOT appear, as the /tf/video_files on the image is shadowed by the bind mount.
# https://docs.docker.com/storage/bind-mounts/#mount-into-a-non-empty-directory-on-the-container
VOLUME /video_files
# If any build steps change the data within the volume after it has been declared, those changes will be discarded.

RUN echo "import tensorflow" \
    && python -c "import tensorflow as tf; print('asserting CUDA'); assert tf.test.is_built_with_cuda();"
# is_built_with_cuda() is not enough; things can go wrong where is_built_with_cuda() is True yet
# the Docker image cannot actually access GPUs later.
# Unfortunately, depending on how the Docker image is built, it can later fail to access the GPU.
# Ideally we would immediately test accessing the GPU immediately as part of the build process.
# Unfortunately, even if the gitlab-runner is set up to pass-through GPU requests,
# the kaniko builder is not.
# So we have to test later after the Docker image is built.

CMD ["python", "-m", "prednet"]

