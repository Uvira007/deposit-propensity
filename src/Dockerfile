# Docker file for Dash app
# Run pipeline once at build so the dashboard has precomputed artifacts
FROM python:3.10-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install ---no-cache-dir -r requirements.txt

# Project files
COPY config/ config/
COPY src/ src/
COPY dashboard/ dashboard/

# Run pipeline once so outputs/ exist (data is downloaded at runtime)
RUN python -m src.run_pipeline

# Expose port (override with PORT env at runtime if needed)
ENV PORT=8050
EXPOSE 8050

# Gunicorn; use PORT so Render can override
CMD ["sh", "-c", "gunicorn dashboard.app:server --bind 0.0.0.0:${PORT:-8050}"]
