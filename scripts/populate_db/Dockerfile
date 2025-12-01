FROM python:3.13-slim-bookworm

# Install system dependencies
RUN apt-get update && apt-get install --no-install-recommends -y \
        build-essential \
        curl \
        libpq-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the script and data files
COPY populate.py .
COPY *.csv ./

# Set HuggingFace cache directory
ENV HF_HOME=/root/.cache/huggingface

# Default entrypoint
ENTRYPOINT ["python", "populate.py"]

# Default command (can be overridden)
CMD ["--help"]

