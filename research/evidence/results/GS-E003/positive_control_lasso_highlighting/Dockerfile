FROM swebench/swesmith.x86_64.pygments_1776_pygments.27649ebb:latest

WORKDIR /workspace
ENV PYTHONPATH=/workspace

# The frozen base image exposes Python 3.12.1 separately from its legacy
# testbed interpreter. Install the test runner into that exact interpreter so
# the prepared runtime, not the legacy default ``python``, is exercised.
RUN /opt/miniconda3/bin/python3.12 -m pip install --disable-pip-version-check --no-cache-dir pytest
