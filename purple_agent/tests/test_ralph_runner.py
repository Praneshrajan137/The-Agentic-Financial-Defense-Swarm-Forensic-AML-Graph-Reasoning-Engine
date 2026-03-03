"""
Test: Ralph Runner -- Per-iteration task executor
PRD Reference: Task A6
"""
import json
import os
import pytest
from pathlib import Path

from src.ralph_runner import (
    load_prd,
    save_prd,
    get_next_task,
    all_tasks_complete,
    log_progress,
)


@pytest.fixture
def prd_path(tmp_path):
    """Create a temporary PRD file for testing."""
    prd = {
        "project": "test",
        "tasks": [
            {"id": "T1", "name": "First", "priority": 1, "deps": [], "status": "pending"},
            {"id": "T2", "name": "Second", "priority": 2, "deps": ["T1"], "status": "pending"},
            {"id": "T3", "name": "Third", "priority": 3, "deps": ["T1"], "status": "pending"},
        ],
    }
    path = tmp_path / "prd.json"
    path.write_text(json.dumps(prd, indent=2))
    return str(path)


@pytest.fixture
def progress_path(tmp_path):
    return str(tmp_path / "progress.txt")


class TestLoadPrd:
    def test_load_prd_reads_valid_file(self, prd_path):
        data = load_prd(prd_path)
        if len(data["tasks"]) != 3:
            raise ValueError(f"Expected 3 tasks, got {len(data['tasks'])}")

    def test_load_prd_raises_on_missing(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_prd(str(tmp_path / "nonexistent.json"))


class TestSavePrd:
    def test_save_prd_atomic_write(self, prd_path):
        data = load_prd(prd_path)
        data["tasks"][0]["status"] = "completed"
        save_prd(prd_path, data)
        reloaded = load_prd(prd_path)
        if reloaded["tasks"][0]["status"] != "completed":
            raise ValueError("Atomic write failed")
        if "last_updated" not in reloaded:
            raise ValueError("last_updated not set")

    def test_save_prd_no_temp_file_left(self, prd_path):
        data = load_prd(prd_path)
        save_prd(prd_path, data)
        parent = Path(prd_path).parent
        tmp_files = list(parent.glob("*.tmp"))
        if tmp_files:
            raise ValueError(f"Temp files left behind: {tmp_files}")


class TestGetNextTask:
    def test_get_next_task_returns_first_pending(self, prd_path):
        data = load_prd(prd_path)
        task = get_next_task(data)
        if task is None or task["id"] != "T1":
            raise ValueError(f"Expected T1, got {task}")

    def test_get_next_task_respects_deps(self, prd_path):
        data = load_prd(prd_path)
        task = get_next_task(data)
        if task["id"] != "T1":
            raise ValueError("T1 should be first (no deps)")
        data["tasks"][0]["status"] = "completed"
        task = get_next_task(data)
        if task is None:
            raise ValueError("Should find T2 or T3 after T1 complete")
        if task["id"] not in ("T2", "T3"):
            raise ValueError(f"Expected T2 or T3, got {task['id']}")

    def test_get_next_task_returns_none_when_all_done(self, prd_path):
        data = load_prd(prd_path)
        for t in data["tasks"]:
            t["status"] = "completed"
        if get_next_task(data) is not None:
            raise ValueError("Should return None when all complete")

    def test_get_next_task_returns_none_when_blocked(self, prd_path):
        data = load_prd(prd_path)
        data["tasks"][0]["status"] = "in_progress"
        task = get_next_task(data)
        if task is not None:
            raise ValueError("T2/T3 depend on T1; should be blocked")


class TestAllTasksComplete:
    def test_all_complete_false(self, prd_path):
        data = load_prd(prd_path)
        if all_tasks_complete(data):
            raise ValueError("Should be False with pending tasks")

    def test_all_complete_true(self, prd_path):
        data = load_prd(prd_path)
        for t in data["tasks"]:
            t["status"] = "completed"
        if not all_tasks_complete(data):
            raise ValueError("Should be True when all completed")


class TestLogProgress:
    def test_log_progress_appends(self, progress_path):
        log_progress(progress_path, "T1", "completed", 1.5)
        log_progress(progress_path, "T2", "completed", 2.3)
        content = Path(progress_path).read_text()
        if "T1" not in content or "T2" not in content:
            raise ValueError("Progress entries missing")
        lines = [l for l in content.strip().split("\n") if l]
        if len(lines) != 2:
            raise ValueError(f"Expected 2 lines, got {len(lines)}")
