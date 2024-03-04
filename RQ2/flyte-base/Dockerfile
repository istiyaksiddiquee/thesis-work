FROM python:3.11.7-slim-bookworm

WORKDIR /root
ENV VENV /opt/venv
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8
ENV PYTHONPATH /root

RUN apt-get update && apt-get install -y build-essential

ENV VENV /opt/venv
RUN python3 -m venv ${VENV}
ENV PATH="${VENV}/bin:$PATH"

RUN pip install flytekit==1.9.1 

COPY requirements.txt /root

RUN pip install -r /root/requirements.txt

COPY . /root

ARG tag
ENV FLYTE_INTERNAL_IMAGE $tag

# docker build -t istiyaksiddiquee/flyte-for-thesis:RQ2RUN1 --build-arg="tag=istiyaksiddiquee/flyte-for-thesis:RQ2RUN1" .
# docker push istiyaksiddiquee/flyte-for-thesis:RQ2RUN1
# sudo ctr image pull docker.io/istiyaksiddiquee/flyte-for-thesis:RQ2RUN1
# pyflyte run --remote --image istiyaksiddiquee/flyte-for-thesis:RQ2RUN1 script.py main_wf