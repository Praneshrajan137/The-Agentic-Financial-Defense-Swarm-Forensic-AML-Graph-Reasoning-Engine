# Publish Green Agent to GitHub Container Registry (for AgentBeats submission)
# Run from repo root. Requires: Docker Desktop running, and one-time: docker login ghcr.io

$ErrorActionPreference = "Stop"
$image = "ghcr.io/praneshrajan137/green-financial-crime-agent:latest"

Write-Host "Checking Docker..." -ForegroundColor Cyan
docker info 2>$null | Out-Null
if ($LASTEXITCODE -ne 0) {
    Write-Host "Docker is not running. Start Docker Desktop and run this script again." -ForegroundColor Red
    exit 1
}

Write-Host "Building $image ..." -ForegroundColor Cyan
docker build -t $image -f Dockerfile .
if ($LASTEXITCODE -ne 0) { exit 1 }

Write-Host "Pushing $image ..." -ForegroundColor Cyan
docker push $image
if ($LASTEXITCODE -ne 0) {
    Write-Host "Push failed. Log in first: docker login ghcr.io -u Praneshrajan137" -ForegroundColor Yellow
    exit 1
}

Write-Host "Done. Use this in the AgentBeats form: $image" -ForegroundColor Green
