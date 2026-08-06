from pathlib import Path


APP_PATH = Path(__file__).parents[1] / "api" / "app.py"


def test_daily_monitor_job_uses_supported_seconds_trigger():
    source = APP_PATH.read_text()

    assert 'trigger=IntervalTrigger(seconds=86400)' in source
    assert 'trigger=IntervalTrigger(days=1)' not in source
