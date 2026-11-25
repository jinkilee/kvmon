ARG VLLM_IMAGE=vllm/vllm-openai:v0.33.0
FROM ${VLLM_IMAGE}

RUN pip install --no-cache-dir nvidia-ml-py prometheus_client

WORKDIR /app
COPY exporter.py entrypoint.sh /app/
RUN chmod +x /app/entrypoint.sh

# NVML and CUDA must agree on GPU numbering so rank N maps to GPU N
ENV CUDA_DEVICE_ORDER=PCI_BUS_ID \
    VLLM_LOG_PATH=/var/log/vllm/vllm.log \
    PYTHONUNBUFFERED=1

# 8000: vLLM OpenAI API + /metrics, 9400: GPU memory exporter
EXPOSE 8000 9400
ENTRYPOINT ["/app/entrypoint.sh"]
