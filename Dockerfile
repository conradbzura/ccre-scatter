FROM python:3.13-slim

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

WORKDIR /app

# Install dependencies (cached unless lockfile changes)
COPY pyproject.toml uv.lock ./
RUN uv sync --no-dev --no-install-project

# Copy application code
COPY app.py scatterplot_helpers.py monitor.sh ./
RUN chmod +x monitor.sh

# marimo default port
EXPOSE 2718

CMD ["sh", "-c", "\
  uv run --no-project marimo run app.py --host 0.0.0.0 & \
  APP_PID=$! && \
  sleep 2 && \
  ./monitor.sh $APP_PID & \
  wait $APP_PID \
"]
