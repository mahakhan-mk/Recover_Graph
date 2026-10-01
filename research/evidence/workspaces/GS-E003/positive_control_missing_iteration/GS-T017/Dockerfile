FROM python:3.12.1-slim

WORKDIR /workspace

RUN pip install --no-cache-dir \
    "pytest==8.3.5" \
    "pygments==2.18.0" \
    "markdown-it-py==3.0.0" \
    "mdurl==0.1.2" \
    "attrs==24.2.0"

ENV PYTHONPATH=/workspace
