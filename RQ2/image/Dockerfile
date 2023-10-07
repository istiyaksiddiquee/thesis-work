FROM python:3.10-slim-buster

WORKDIR /root
ENV PYTHONPATH /root

ARG VERSION
ARG DOCKER_IMAGE

RUN apt-get update && apt-get install build-essential -y

# Pod tasks should be exposed in the default image
RUN pip install flytekit==1.9.1 \
	flytekitplugins-pod==1.9.1 \
	flytekitplugins-deck-standard==1.9.1

COPY ./requirements.txt .

RUN pip install -r requirements.txt

COPY ./script.py .

RUN useradd -u 1000 flytekit
RUN chown flytekit: /root
USER flytekit

# ENV FLYTE_INTERNAL_IMAGE "$DOCKER_IMAGE"