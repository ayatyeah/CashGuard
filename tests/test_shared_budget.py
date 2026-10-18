"""The server lowers a running job's deadline when a batch of videos arrives; the analyzer must follow it."""
import time

from app.analyzers.ml import MLAnalyzer


def test_no_file_keeps_budget(monkeypatch):
    monkeypatch.delenv("FORTE_DEADLINE_FILE", raising=False)
    assert MLAnalyzer.shared_budget(time.time(), 200.0) == 200.0


def test_file_shrinks_but_never_extends(tmp_path, monkeypatch):
    f = tmp_path / "job.deadline"
    monkeypatch.setenv("FORTE_DEADLINE_FILE", str(f))
    t0 = time.time()
    f.write_text(f"{t0 + 75:.3f}")          # 75 s left for the whole job → 60 s for analysis (15 s reserve)
    assert abs(MLAnalyzer.shared_budget(t0, 200.0) - 60.0) < 0.5
    f.write_text(f"{t0 + 1000:.3f}")        # a later deadline never extends the original budget
    assert MLAnalyzer.shared_budget(t0, 200.0) == 200.0
    f.write_text("garbage")                 # unreadable → keep what we had
    assert MLAnalyzer.shared_budget(t0, 120.0) == 120.0
