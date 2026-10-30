"""The eval dashboard's API (/admin/evals*) and the voice speaking indicator events."""

import json
import os

import pytest
from fastapi.testclient import TestClient

from app.main import create_app
from tests.agent_helpers import ScriptedLLM, say
from tests.test_voice_session import start
from tests.voice_helpers import noise, quiet


def _write(base, name, run_id, mtime, success):
    folder = base / name
    folder.mkdir()
    path = folder / "results.json"
    data = {"run_id": run_id, "started_at": "2026-10-30T10:00:00+00:00", "agent_model": "m",
            "summary": {"runs": 3, "task_success_rate": success}, "scenarios": []}  # fmt: skip
    path.write_text(json.dumps(data))
    os.utime(path, (mtime, mtime))


@pytest.mark.db
def test_eval_results_are_listed_newest_first_and_served(test_db_settings, tmp_path):
    _write(tmp_path, "older", "r1", 1_000, 0.5)
    _write(tmp_path, "newer", "r2", 2_000, 0.9)
    (tmp_path / "broken").mkdir()
    (tmp_path / "broken" / "results.json").write_text("{not json")
    os.utime(tmp_path / "broken" / "results.json", (500, 500))
    settings = test_db_settings.model_copy(
        update={"gemini_api_key": None, "eval_results_dir": tmp_path}
    )
    with TestClient(create_app(settings)) as client:
        runs = client.get("/admin/evals").json()
        assert [r["name"] for r in runs] == ["newer", "older"]  # unreadable files skipped
        assert runs[0]["task_success_rate"] == 0.9
        latest = client.get("/admin/evals/latest").json()
        assert (latest["name"], latest["run_id"]) == ("newer", "r2")
        assert client.get("/admin/evals/older").json()["run_id"] == "r1"
        for bad in ("missing", "..", "%2e%2e%2fsecrets", ".hidden"):
            assert client.get(f"/admin/evals/{bad}").status_code == 404


@pytest.mark.db
def test_no_eval_results_is_a_404_with_a_hint(test_db_settings, tmp_path):
    settings = test_db_settings.model_copy(
        update={"gemini_api_key": None, "eval_results_dir": tmp_path / "none"}
    )
    with TestClient(create_app(settings)) as client:
        assert client.get("/admin/evals").json() == []
        response = client.get("/admin/evals/latest")
        assert response.status_code == 404 and "make eval-smoke" in response.json()["detail"]


@pytest.mark.anyio
@pytest.mark.db
async def test_the_session_says_when_the_agent_starts_and_stops_being_heard(tx_db):
    llm = ScriptedLLM(say("Your order is on its way."))
    session, driver, _, _ = await start(tx_db, llm, script=["where is my order"])
    await driver.play(noise(800) + quiet(700))
    await driver.settle()
    await session.close()
    audible = [e["playing"] for e in driver.events if e["type"] == "agent_audio"]
    assert audible[:2] == [True, False] and audible[-1] is False
