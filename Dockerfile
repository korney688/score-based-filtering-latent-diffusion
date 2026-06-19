FROM pytorch/pytorch:2.7.1-cuda11.8-cudnn9-runtime

ENV PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg \
    PIP_NO_CACHE_DIR=1

WORKDIR /workspace

RUN apt-get update && apt-get install -y --no-install-recommends \
    bash \
    ca-certificates \
    curl \
    git \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /workspace/requirements.txt

RUN python -m pip install --upgrade pip \
    && python -m pip install -r requirements.txt

COPY configs/ /workspace/configs/
COPY scripts/ /workspace/scripts/
COPY src/ /workspace/src/
COPY docs/ /workspace/docs/
COPY tests/ /workspace/tests/
COPY README.md /workspace/README.md

RUN mkdir -p /workspace/data /workspace/checkpoints /workspace/experiments /workspace/outputs /workspace/logs \
    && python -m compileall -q scripts src

CMD ["bash"]
