# syntax=docker/dockerfile:1.6

FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PIP_NO_CACHE_DIR=1 \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    SHELL=/bin/bash

ARG USERNAME=dev
ARG USER_UID=1000
ARG USER_GID=$USER_UID

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        git \
        openssh-client \
        sudo \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --gid "${USER_GID}" "${USERNAME}" \
    && useradd --uid "${USER_UID}" --gid "${USER_GID}" -m "${USERNAME}" \
    && echo "${USERNAME} ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers.d/${USERNAME}

WORKDIR /workspace

RUN pip install --upgrade pip \
    && pip install hatch

USER ${USERNAME}

SHELL ["/bin/bash", "-c"]
CMD ["bash"]
