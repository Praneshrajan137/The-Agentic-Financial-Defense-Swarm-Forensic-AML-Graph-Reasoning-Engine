#!/usr/bin/env bash
# Publish Green Agent to GitHub Container Registry (for AgentBeats submission)
# Run from repo root. Requires: Docker running, and one-time: docker login ghcr.io

set -e
IMAGE="ghcr.io/praneshrajan137/green-financial-crime-agent:latest"

echo "Checking Docker..."
docker info >/dev/null 2>&1 || { echo "Docker is not running. Start Docker and run this script again."; exit 1; }

echo "Building $IMAGE ..."
docker build -t "$IMAGE" -f Dockerfile .

echo "Pushing $IMAGE ..."
docker push "$IMAGE" || { echo "Push failed. Log in first: docker login ghcr.io -u Praneshrajan137"; exit 1; }

echo "Done. Use this in the AgentBeats form: $IMAGE"
