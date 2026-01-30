FROM python:3.10-slim

WORKDIR /app

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    GREEN_AGENT_URL=http://localhost:8000 \
    PARTICIPANT_ID=baseline_purple_agent

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy agent code
COPY src/ ./src/

# Default command runs the baseline agent
# Arguments can be overridden via docker-compose or docker run
ENTRYPOINT ["python", "src/baseline_agent.py"]
CMD ["--green-url", "http://green-agent:8000", "--nodes", "0", "42", "100"]
