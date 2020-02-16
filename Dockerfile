ARG BASE_IMAGE
ARG HTTP_PROXY
ARG HTTPS_PROXY
FROM $BASE_IMAGE

# Using the tensorflow base image, the current directory is /tf.
# That is where Jupyter will open.
# Jupyter will display it as simply / ,
# but checking the current working directory in code will confimr it is /tf.

# .dockerignore applies to COPY
COPY ./ ./prednet
COPY ./src/prednet/tests/resources/ ./test_videos

ENV http_proxy=$HTTP_PROXY
ENV https_proxy=$HTTPS_PROXY

RUN pip uninstall --yes enum34
RUN pip install --no-cache ./prednet

RUN mkdir ./video_files
RUN touch ./video_files/dummyfile
VOLUME /video_files
# If any build steps change the data within the volume after it has been declared, those changes will be discarded.
