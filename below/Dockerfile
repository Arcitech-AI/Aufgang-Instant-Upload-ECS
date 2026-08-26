FROM public.ecr.aws/lambda/python:3.12

WORKDIR ${LAMBDA_TASK_ROOT}

# Docling depends on torch; the default PyPI Linux wheel is often CUDA-enabled (~GB).
# Lambda is CPU-only — install torch/torchvision from PyTorch's CPU index first so
# `pip install docling` does not replace them with a huge CUDA build.
RUN pip install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cpu \
    "torch>=2.2.2,<3" "torchvision>=0.0.0,<1"

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY script.sh ${LAMBDA_TASK_ROOT}/script.sh
RUN chmod 755 ${LAMBDA_TASK_ROOT}/script.sh

COPY app.py ${LAMBDA_TASK_ROOT}/app.py

CMD ["app.handler"]
