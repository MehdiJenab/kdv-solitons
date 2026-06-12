FROM python:3.12-slim

WORKDIR /app

# Install the package with its web extra (Flask).
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir ".[web]"

# App entry point and templates.
COPY simple_web_app.py ./
COPY templates ./templates

EXPOSE 5001
CMD ["python", "simple_web_app.py"]
