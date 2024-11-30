FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN curl -sSL https://install.python-poetry.org | python3 -

# Set working directory
WORKDIR /app

# Copy Poetry files
COPY pyproject.toml poetry.lock ./

# Copy project files
COPY . .

# Configure Poetry
RUN poetry config virtualenvs.create false \
    && poetry install --no-interaction --no-ansi

# Create necessary directories
RUN mkdir -p logs output src/grant_finder/data/company_docs src/grant_finder/data/funding_sources

# Copy example configs if not exist
RUN cp -n src/grant_finder/config/user_config.yaml.example src/grant_finder/config/user_config.yaml || true
RUN cp -n src/grant_finder/data/funding_sources/sources.csv.example src/grant_finder/data/funding_sources/sources.csv || true

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Default command
CMD ["poetry", "run", "python", "-m", "grant_finder.main"]