import os
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_ENV_PATH = REPO_ROOT / ".env"


def load_dotenv(env_path=DEFAULT_ENV_PATH):
    values = {}
    if not env_path.exists():
        return values

    for raw_line in env_path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        if "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()

        if value and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]

        values[key] = value

    return values


def get_config_value(name, cli_value=None):
    if cli_value:
        return cli_value

    env_values = load_dotenv()
    if name in os.environ:
        return os.environ[name]
    return env_values.get(name)


def require_config_value(name, cli_value=None):
    value = get_config_value(name=name, cli_value=cli_value)
    if value:
        return value

    raise ValueError(
        f"Missing required path configuration '{name}'. "
        f"Set it in .env, export it in the shell, or pass the matching CLI argument."
    )
