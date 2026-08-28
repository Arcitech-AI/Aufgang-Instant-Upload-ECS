# syntax=docker/dockerfile:1

FROM public.ecr.aws/lambda/python:3.12 AS builder

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /build
COPY requirements-cpu.txt requirements.txt ./

RUN python -m venv /opt/venv \
    && /opt/venv/bin/python -m pip install --requirement requirements-cpu.txt \
    && /opt/venv/bin/python -m pip install --requirement requirements.txt \
    && /opt/venv/bin/python -m pip check \
    && /opt/venv/bin/python -m pip uninstall --yes pip

FROM public.ecr.aws/lambda/python:3.12 AS runtime

COPY --from=builder /opt/venv/lib/python3.12/site-packages/ ${LAMBDA_TASK_ROOT}/
COPY main.py ${LAMBDA_TASK_ROOT}/

CMD [ "main.handler" ]
