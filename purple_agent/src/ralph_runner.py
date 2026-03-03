"""
Ralph Runner -- Per-iteration task executor for Ralph Wiggum loop
PRD Reference: Task A6

Each invocation:
1. Loads prd.json (task list)
2. Finds next pending task (lowest priority)
3. Marks it in_progress, saves atomically
4. Logs to progress.txt
5. Exits 0 if all done, 1 if more remain
"""
import argparse
import json
import logging
import os
import sys
import tempfile
import time
from typing import Any

logger = logging.getLogger(__name__)


def load_prd(path: str) -> dict[str, Any]:
    """Load PRD task list from JSON file."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"PRD file not found: {path}")
    with open(path, "r") as f:
        return json.load(f)


def save_prd(path: str, data: dict[str, Any]) -> None:
    """Atomically save PRD task list (Rule 23: tempfile + fsync + replace)."""
    data["last_updated"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    dir_name = os.path.dirname(os.path.abspath(path))
    fd = tempfile.NamedTemporaryFile(
        "w", dir=dir_name, delete=False, suffix=".tmp"
    )
    try:
        json.dump(data, fd, indent=2)
        fd.flush()
        os.fsync(fd.fileno())
        fd.close()
        os.replace(fd.name, path)
    except Exception:
        fd.close()
        if os.path.exists(fd.name):
            os.unlink(fd.name)
        raise


def get_next_task(prd: dict[str, Any]) -> dict[str, Any] | None:
    """Return the next pending task with lowest priority, or None if all done."""
    tasks = prd.get("tasks", [])
    pending = [t for t in tasks if t.get("status") == "pending"]
    if not pending:
        return None

    pending.sort(key=lambda t: t.get("priority", 999))

    for candidate in pending:
        deps = candidate.get("deps", [])
        all_deps_done = all(
            any(
                t["id"] == dep_id and t.get("status") in ("completed", "done")
                for t in tasks
            )
            for dep_id in deps
        )
        if all_deps_done:
            return candidate
    return None


def all_tasks_complete(prd: dict[str, Any]) -> bool:
    """Check if all tasks are completed."""
    tasks = prd.get("tasks", [])
    return all(t.get("status") in ("completed", "done") for t in tasks)


def log_progress(path: str, task_id: str, status: str, duration: float) -> None:
    """Append progress entry to progress.txt."""
    timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    line = f"[{timestamp}] Task {task_id}: {status} ({duration:.1f}s)\n"
    with open(path, "a") as f:
        f.write(line)


def main() -> int:
    """Run one iteration of the Ralph Wiggum loop.

    Returns 0 if all tasks complete, 1 if more tasks remain.
    """
    parser = argparse.ArgumentParser(description="Ralph Runner -- single iteration")
    parser.add_argument("--state-dir", default="./state")
    parser.add_argument("--prd", default="plans/prd.json")
    parser.add_argument("--progress", default="progress.txt")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    prd = load_prd(args.prd)

    if all_tasks_complete(prd):
        logger.info("All tasks already complete.")
        return 0

    task = get_next_task(prd)
    if task is None:
        logger.warning("No actionable tasks (dependencies not met).")
        return 1

    task_id = task["id"]
    start = time.time()
    logger.info("Starting task %s: %s", task_id, task.get("name", ""))

    task["status"] = "in_progress"
    save_prd(args.prd, prd)

    task["status"] = "completed"
    duration = time.time() - start
    save_prd(args.prd, prd)

    log_progress(args.progress, task_id, "completed", duration)
    logger.info("Task %s completed in %.1fs", task_id, duration)

    if all_tasks_complete(prd):
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
