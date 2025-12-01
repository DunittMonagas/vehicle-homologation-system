# Database Population Script

This script populates the PostgreSQL and Upstash Vector databases with vehicle data from a CSV file.

## Features

- **Batch Processing**: Efficiently processes large CSV files using configurable batch sizes
- **HuggingFace Embeddings**: Uses local HuggingFace models (default: `all-MiniLM-L6-v2`) for embedding calculation
- **Dual Database Support**: Populates both PostgreSQL (vehicle records) and Upstash Vector (embeddings)
- **Flexible Options**: Skip PostgreSQL or Vector population independently
- **Docker Support**: Run the script in an isolated container environment

## Prerequisites

1. PostgreSQL database must be running and accessible
2. Database tables must already be created (run Alembic migrations first)
3. Upstash Vector index must be created with appropriate dimensions (384 for `all-MiniLM-L6-v2`)
4. Required environment variables must be set

## Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `DB_USER` | PostgreSQL username | Yes |
| `DB_PASSWORD` | PostgreSQL password | Yes |
| `DB_NAME` | PostgreSQL database name | Yes |
| `DB_HOST` | PostgreSQL host | No (default: `localhost`) |
| `DB_PORT` | PostgreSQL port | No (default: `5432`) |
| `UPSTASH_VECTOR_REST_URL` | Upstash Vector REST API URL | Yes |
| `UPSTASH_VECTOR_REST_TOKEN` | Upstash Vector REST API token | Yes |
| `EMBEDDING_MODEL` | HuggingFace model name | No (default: `all-MiniLM-L6-v2`) |

## CSV Format

The script expects a CSV file with the following columns:

```csv
versionc,id_crabi
"FIAT MOBI 2024 TREKKING, L4, 1.0L, 69 CP, 5 PUERTAS, AUT",FM-100
"HYUNDAI TUCSON 2023 LIMITED, L4, 1.6T, 226 CP, 5 PUERTAS, AUT, HEV",HT-200
```

- `versionc`: Vehicle description (used for embedding calculation)
- `id_crabi`: Unique vehicle identifier

## Usage

### Option 1: Using Docker (Recommended)

#### 1. Place your CSV file in the script folder

Copy your CSV file to `scripts/populate_db/`:

```bash
cp /path/to/your/vehicles.csv scripts/populate_db/
```

#### 2. Create an environment file

Create a `.env` file in the script folder with your configuration:

```bash
DB_USER=your_db_user
DB_PASSWORD=your_db_password
DB_NAME=your_db_name
DB_HOST=host.docker.internal  # Use this to connect to host's localhost
DB_PORT=5432
UPSTASH_VECTOR_REST_URL=https://your-index.upstash.io
UPSTASH_VECTOR_REST_TOKEN=your_token
EMBEDDING_MODEL=all-MiniLM-L6-v2
```

#### 3. Build the Docker image

```bash
cd scripts/populate_db
docker build -t populate-db .
```

#### 4. Run the script

```bash
# Basic usage - populate both databases
docker run --rm \
    --env-file .env \
    -v populate_db_hf_cache:/root/.cache/huggingface \
    populate-db --csv vehicles.csv

# Skip PostgreSQL (only populate vectors)
docker run --rm \
    --env-file .env \
    -v populate_db_hf_cache:/root/.cache/huggingface \
    populate-db --csv vehicles.csv --skip-postgres

# Skip vectors (only populate PostgreSQL)
docker run --rm \
    --env-file .env \
    populate-db --csv vehicles.csv --skip-vectors

# Custom batch sizes
docker run --rm \
    --env-file .env \
    -v populate_db_hf_cache:/root/.cache/huggingface \
    populate-db --csv vehicles.csv \
        --postgres-batch-size 200 \
        --vector-batch-size 25

# Use a specific namespace for vectors
docker run --rm \
    --env-file .env \
    -v populate_db_hf_cache:/root/.cache/huggingface \
    populate-db --csv vehicles.csv --namespace my-namespace
```

### Option 2: Using Docker Compose

You can integrate this script with the main application's Docker Compose:

```yaml
# Add to your docker-compose.yaml
services:
  populate-db:
    build:
      context: ./scripts/populate_db
    env_file: .env
    environment:
      - DB_HOST=db  # Use the service name for internal Docker network
      - HF_HOME=/root/.cache/huggingface
    volumes:
      - hf_cache:/root/.cache/huggingface
    depends_on:
      db:
        condition: service_healthy
    command: ["--csv", "vehicles.csv"]
    profiles:
      - populate  # Only run when explicitly requested
```

Run with:

```bash
# Run with the populate profile
docker-compose --profile populate run --rm populate-db
```

### Option 3: Running Locally (without Docker)

#### 1. Install dependencies

```bash
cd scripts/populate_db
pip install -r requirements.txt
```

#### 2. Set environment variables

```bash
export DB_USER=your_db_user
export DB_PASSWORD=your_db_password
export DB_NAME=your_db_name
export DB_HOST=localhost
export DB_PORT=5432
export UPSTASH_VECTOR_REST_URL=https://your-index.upstash.io
export UPSTASH_VECTOR_REST_TOKEN=your_token
export EMBEDDING_MODEL=all-MiniLM-L6-v2
```

#### 3. Run the script

```bash
python populate.py --csv /path/to/vehicles.csv
```

## Command Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--csv` | Path to the CSV file (required) | - |
| `--postgres-batch-size` | Batch size for PostgreSQL inserts | 100 |
| `--vector-batch-size` | Batch size for vector embeddings | 50 |
| `--namespace` | Namespace for vector database | None |
| `--skip-postgres` | Skip PostgreSQL population | False |
| `--skip-vectors` | Skip vector database population | False |

## Notes

### First Run Performance

The first run will download the HuggingFace model, which may take some time depending on your internet connection. Subsequent runs will use the cached model.

Using a Docker volume for the HuggingFace cache (`-v populate_db_hf_cache:/root/.cache/huggingface`) ensures the model is persisted between runs.

### Embedding Dimensions

The default model `all-MiniLM-L6-v2` produces 384-dimensional embeddings. Make sure your Upstash Vector index is configured with the same dimensions.

### Network Configuration

When running in Docker and connecting to a local PostgreSQL:
- On Linux: Use `--network host` or the host IP
- On macOS/Windows: Use `host.docker.internal` as `DB_HOST`

### Error Handling

The script will:
- Validate all required environment variables before processing
- Skip invalid CSV rows with a warning
- Rollback PostgreSQL transactions on error
- Report detailed progress and errors in the logs

