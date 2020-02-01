ARG BASE_IMAGE
ARG HTTP_PROXY
ARG HTTPS_PROXY
FROM $BASE_IMAGE

# .dockerignore applies to COPY
COPY ./ .

ENV http_proxy=$HTTP_PROXY
ENV https_proxy=$HTTPS_PROXY

RUN pip uninstall --yes enum34
RUN pip install --no-cache .
