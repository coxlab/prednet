ARG BASE_IMAGE
ARG HTTP_PROXY
ARG HTTPS_PROXY
FROM $BASE_IMAGE

# .dockerignore applies to COPY
COPY ./ ./prednet

ENV http_proxy=$HTTP_PROXY
ENV https_proxy=$HTTPS_PROXY

RUN pip uninstall --yes enum34
RUN pip install --no-cache .

RUN mkdir /video_files
VOLUME /video_files
# If any build steps change the data within the volume after it has been declared, those changes will be discarded.
