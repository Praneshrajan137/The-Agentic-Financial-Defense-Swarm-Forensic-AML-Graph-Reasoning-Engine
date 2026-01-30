# AgentBeats Green Agent Submission

## Docker Image for Submission

Use this **fully qualified image** in the AgentBeats submission form (Docker Image field):

```
ghcr.io/praneshrajan137/green-financial-crime-agent:latest
```

- **Registry**: `ghcr.io` (GitHub Container Registry)
- **Repository**: `praneshrajan137/green-financial-crime-agent`
- **Tag**: `latest`

---

## One-Time: Publish the Image to GitHub Container Registry

Before submitting, push the Green Agent image so AgentBeats can pull it.

**Quick way (after Docker is running and you've logged in once):**

```powershell
# From repo root, Windows:
.\scripts\publish-green-agent.ps1
```

```bash
# From repo root, Linux/macOS:
./scripts/publish-green-agent.sh
```

### 1. Log in to GitHub Container Registry (one-time)

```bash
# Create a Personal Access Token (PAT) at:
# GitHub → Settings → Developer settings → Personal access tokens
# Required scope: write:packages (and read:packages if you want to pull private images)

echo YOUR_GITHUB_PAT | docker login ghcr.io -u YOUR_GITHUB_USERNAME --password-stdin
```

Replace `YOUR_GITHUB_PAT` and `YOUR_GITHUB_USERNAME` (e.g. `Praneshrajan137`).

### 2. Build and tag the Green Agent image

From the repository root:

```bash
docker build -t ghcr.io/praneshrajan137/green-financial-crime-agent:latest -f Dockerfile .
```

### 3. Push to GHCR

```bash
docker push ghcr.io/praneshrajan137/green-financial-crime-agent:latest
```

### 4. (Optional) Make the package public

- Go to **GitHub → Your profile → Packages → green-financial-crime-agent**
- **Package settings → Change visibility → Public**

---

## Verify locally

```bash
docker pull ghcr.io/praneshrajan137/green-financial-crime-agent:latest
docker run -p 8000:8000 ghcr.io/praneshrajan137/green-financial-crime-agent:latest \
  python main.py serve --host 0.0.0.0 --port 8000 --generate-on-startup --seed 42 --difficulty 5 --output-dir /tmp/out
# In another terminal:
curl http://localhost:8000/health
```

---

## If you use Docker Hub instead

If you prefer Docker Hub, use this image name in the form:

```
docker.io/YOUR_DOCKERHUB_USERNAME/green-financial-crime-agent:latest
```

Build and push:

```bash
docker build -t YOUR_DOCKERHUB_USERNAME/green-financial-crime-agent:latest -f Dockerfile .
docker push YOUR_DOCKERHUB_USERNAME/green-financial-crime-agent:latest
```

Then submit: `docker.io/YOUR_DOCKERHUB_USERNAME/green-financial-crime-agent:latest`
