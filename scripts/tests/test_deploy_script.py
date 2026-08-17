from pathlib import Path


def test_all_deploy_compose_calls_use_runtime_env_file() -> None:
    script = Path(__file__).resolve().parents[1].joinpath("deploy.sh").read_text()

    compose_calls = [
        line.strip()
        for line in script.splitlines()
        if "docker compose" in line and not line.strip().startswith("#")
    ]

    assert compose_calls
    assert all('--env-file "$ENV_FILE"' in line for line in compose_calls), compose_calls