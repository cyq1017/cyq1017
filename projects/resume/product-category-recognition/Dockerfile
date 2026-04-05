FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git wget curl && \
    rm -rf /var/lib/apt/lists/*

# Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project
COPY . .

# Pre-download model (cached in image)
RUN python -c "from transformers import Blip2Processor, Blip2ForConditionalGeneration; \
    Blip2Processor.from_pretrained('Salesforce/blip2-opt-2.7b'); \
    Blip2ForConditionalGeneration.from_pretrained('Salesforce/blip2-opt-2.7b')"

CMD ["make", "all"]
