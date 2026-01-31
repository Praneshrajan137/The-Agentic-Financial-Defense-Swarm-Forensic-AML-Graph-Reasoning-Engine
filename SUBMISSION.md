# AgentBeats Green Agent Submission

## Docker Image for Submission

Use this **fully qualified image** in the AgentBeats submission form (Docker Image field):

```
praneshrajan15/green-financial-crime-agent:latest
```

- **Registry**: Docker Hub
- **Repository**: `praneshrajan15/green-financial-crime-agent`
- **Tag**: `latest`

---

## How the Image is Published

The image is built and pushed automatically by GitHub Actions on every push to `main`/`master`:

1. **Push to the repo** triggers the workflow
2. Workflow builds and pushes to Docker Hub
3. Image is publicly available at `praneshrajan15/green-financial-crime-agent:latest`

See [`.github/workflows/publish-dockerhub.yml`](.github/workflows/publish-dockerhub.yml) for the workflow configuration.

---

## Verify the Image

```bash
# Pull the image
docker pull praneshrajan15/green-financial-crime-agent:latest

# Run the Green Agent
docker run -p 8000:8000 praneshrajan15/green-financial-crime-agent:latest \
  python main.py serve --host 0.0.0.0 --port 8000 --generate-on-startup --seed 42 --difficulty 5 --output-dir /tmp/out

# In another terminal, verify health:
curl http://localhost:8000/health
```

---

## Run Full Benchmark End-to-End

```bash
# Clone the repository
git clone https://github.com/Praneshrajan137/The-Agentic-Financial-Defense-Swarm-Forensic-AML-Graph-Reasoning-Engine.git
cd The-Agentic-Financial-Defense-Swarm-Forensic-AML-Graph-Reasoning-Engine

# Run with Docker Compose
docker-compose up --build

# The Green Agent will start on port 8000
# The Purple Agent will connect and run investigation
```

---

## Reproducibility Demonstration

```bash
# Run 3 evaluations with fixed seed
python scripts/run_benchmark.py --seed 42 --difficulty 5 --runs 3 --output results.json
```

Expected: All 3 runs produce identical scores (variance = 0).
