# Database Migrations with Alembic

This directory contains database migrations managed by Alembic.

## Common Commands

### Create a new migration
After modifying SQLAlchemy models in `app/models/`, generate a new migration:

```bash
uv run alembic revision --autogenerate -m "Description of changes"
```

### Apply migrations
Apply all pending migrations:

```bash
uv run alembic upgrade head
```

### Rollback migrations
Rollback the last migration:

```bash
uv run alembic downgrade -1
```

### View migration history
```bash
uv run alembic history
```

### View current migration version
```bash
uv run alembic current
```

## Docker Environment

When using Docker, the database is automatically started with:

```bash
docker compose up -d db
```

The web service will wait for the database to be healthy before starting.

## Configuration

- Database connection is configured in `app/core/config.py`
- Alembic is configured to automatically detect model changes
- All SQLAlchemy models must be imported in `alembic/env.py` for autogenerate to work

