FROM python:3.11.7-slim-bookworm

WORKDIR /work
USER root 

ARG VERSION
ARG DOCKER_IMAGE

RUN apt-get update && apt-get install build-essential -y

# Pod tasks should be exposed in the default image
RUN pip install flytekit==1.9.1 \
	flytekitplugins-pod==1.9.1 \
	flytekitplugins-deck-standard==1.9.1

COPY ./requirements.txt .

RUN pip install -r requirements.txt

COPY ./x_train_val.pickle .
COPY ./y_train_val.pickle .
COPY ./x_test.pickle .
COPY ./y_test.pickle .
COPY ./script.py .
COPY ./rq2_vosoughi_False_features.csv .
COPY ./rq2_vosoughi_True_features.csv .


# ENV FLYTE_INTERNAL_IMAGE "$DOCKER_IMAGE" 