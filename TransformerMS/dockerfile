# ---------- Build stage ----------
FROM python:3.12-slim AS builder

# Set working directory
WORKDIR /app

# Install build dependencies only for this stage
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --upgrade pip

# Copy requirements and install them
COPY requirements.txt .
RUN pip install --prefix=/install -r requirements.txt

# ---------- Final stage ----------
FROM python:3.12-slim AS final

# Set working directory
WORKDIR /app

# Copy only installed packages from builder
COPY --from=builder /install /usr/local

# Copy your application code
COPY . .

# Expose app port
EXPOSE 8082

# Default command to run the app
CMD ["python", "main.py"]
