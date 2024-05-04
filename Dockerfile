ARG PYTHON_VERSION=3.9.2
FROM python:${PYTHON_VERSION}-slim as base

# Expose the port that the application listens on.
EXPOSE 8000

# Prevents Python from writing pyc files.
ENV PYTHONDONTWRITEBYTECODE = 1

# Keeps Python from buffering stdout and stderr to avoid situations where
# the application crashes without emitting any logs due to buffering.
ENV PYTHONUNBUFFERED = 1

WORKDIR /app
COPY . /app

# Install pip req
COPY docker_requirements.txt .
RUN pip install -r docker_requirements.txt 

# Run the application
CMD streamlit run ./prod/docker_client.py --server.port 8081
