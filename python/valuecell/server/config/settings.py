"""Settings configuration for ValueCell Server."""

import os
from functools import lru_cache

from valuecell.config.constants import PROJECT_ROOT
from valuecell.utils.env import get_system_env_dir


def _get_project_root() -> str:
    """Get project root directory path.

    Layout assumption: this file is at repo_root/python/valuecell/server/config/settings.py
    We walk up 4 levels to reach repo_root.
    """
    here = os.path.dirname(__file__)
    repo_root = os.path.abspath(os.path.join(here, "..", "..", "..", ".."))
    return repo_root


def _default_db_path() -> str:
    """Get default database DSN under the system application directory.

    Mirrors `.env` location so the SQLite file lives alongside user-level config:
    - macOS: `~/Library/Application Support/ValueCell/valuecell.db`
    - Linux: `~/.config/valuecell/valuecell.db`
    - Windows: `%APPDATA%\\ValueCell\\valuecell.db`
    """
    system_dir = get_system_env_dir()
    return f"sqlite:///{os.path.join(str(system_dir), 'valuecell.db')}"


_SUPPORTED_MARKET_DATA_PROVIDERS = ("okx", "binance", "gate", "mexc")

_SUPPORTED_MARKET_INTERVALS = frozenset({"1m", "3m", "5m", "15m", "30m", "1h", "4h", "1d"})
_SUPPORTED_MARKET_SYMBOLS = frozenset(
    {
        "BTC-USDT",
        "ETH-USDT",
        "BNB-USDT",
        "SOL-USDT",
        "XRP-USDT",
        "ADA-USDT",
        "DOGE-USDT",
        "DOT-USDT",
        "USDC-USDT",
        "LTC-USDT",
        "BCH-USDT",
        "LINK-USDT",
        "AVAX-USDT",
        "MATIC-USDT",
        "POL-USDT",
        "UNI-USDT",
        "ATOM-USDT",
        "ETC-USDT",
        "FIL-USDT",
        "AAVE-USDT",
        "SAND-USDT",
        "MANA-USDT",
        "ALGO-USDT",
        "FTM-USDT",
        "NEAR-USDT",
        "GRT-USDT",
        "CAKE-USDT",
        "XLM-USDT",
        "EOS-USDT",
        "TRX-USDT",
        "WBTC-USDT",
        "ARB-USDT",
        "OP-USDT",
        "MKR-USDT",
        "SNX-USDT",
        "CRV-USDT",
        "1INCH-USDT",
        "KAVA-USDT",
        "ZRX-USDT",
        "BAT-USDT",
        "OMG-USDT",
        "QTUM-USDT",
        "ICX-USDT",
        "VET-USDT",
        "THETA-USDT",
        "NEO-USDT",
        "ONT-USDT",
        "ZIL-USDT",
        "RVN-USDT",
        "DASH-USDT",
        "HBAR-USDT",
        "IOTA-USDT",
        "WAVES-USDT",
        "KSM-USDT",
        "RSR-USDT",
        "CELR-USDT",
        "FET-USDT",
        "OCEAN-USDT",
        "REQ-USDT",
        "BNT-USDT",
        "LRC-USDT",
        "GNO-USDT",
        "PAXG-USDT",
        "UMA-USDT",
        "BAL-USDT",
        "SPELL-USDT",
        "AUDIO-USDT",
        "RAY-USDT",
        "CELO-USDT",
        "MASK-USDT",
        "COTI-USDT",
        "CHZ-USDT",
        "ENJ-USDT",
        "GAS-USDT",
        "HOT-USDT",
        "IOST-USDT",
        "KEY-USDT",
        "LOKA-USDT",
        "MBL-USDT",
        "NKN-USDT",
        "OAX-USDT",
        "RIF-USDT",
        "SXP-USDT",
    }
)


def _parse_default_market_symbols(value: str) -> tuple[str, ...]:
    symbols = tuple(item.strip().upper().replace("/", "-") for item in value.split(",") if item.strip())
    if not symbols:
        raise ValueError("VALUECELL_MARKET_DEFAULT_SYMBOLS must include a symbol")
    invalid = sorted(set(symbols) - _SUPPORTED_MARKET_SYMBOLS)
    if invalid:
        raise ValueError(
            "VALUECELL_MARKET_DEFAULT_SYMBOLS contains unsupported symbols: "
            + ", ".join(invalid)
        )
    return tuple(dict.fromkeys(symbols))


def _default_market_interval(value: str) -> str:
    interval = value.strip().lower()
    if interval not in _SUPPORTED_MARKET_INTERVALS:
        raise ValueError(f"VALUECELL_MARKET_DEFAULT_INTERVAL has unsupported interval: {value}")
    return interval


def _parse_market_data_providers(value: str) -> tuple[str, ...]:
    """Parse and validate public OHLCV provider fallback order."""
    providers = tuple(item.strip().lower() for item in value.split(",") if item.strip())
    if not providers:
        raise ValueError("VALUECELL_MARKET_DATA_PROVIDERS must include a provider")
    invalid = sorted(set(providers) - set(_SUPPORTED_MARKET_DATA_PROVIDERS))
    if invalid:
        raise ValueError(
            "VALUECELL_MARKET_DATA_PROVIDERS contains unsupported providers: "
            + ", ".join(invalid)
        )
    return tuple(dict.fromkeys(providers))


def _positive_int_env(name: str, default: int) -> int:
    value = int(os.getenv(name, str(default)))
    if value < 1:
        raise ValueError(f"{name} must be at least 1")
    return value


def _positive_float_env(name: str, default: float) -> float:
    value = float(os.getenv(name, str(default)))
    if value <= 0:
        raise ValueError(f"{name} must be greater than 0")
    return value

class Settings:
    """Server configuration settings."""

    def __init__(self):
        """Initialize settings from environment variables."""
        # Application Configuration
        self.APP_NAME = os.getenv("APP_NAME", "ValueCell Server")
        self.APP_VERSION = os.getenv("APP_VERSION", "0.1.0")
        self.APP_ENVIRONMENT = os.getenv("APP_ENVIRONMENT", "development")

        # API Configuration
        self.API_HOST = os.getenv("API_HOST", "0.0.0.0")
        self.API_PORT = int(os.getenv("API_PORT", "8000"))
        self.API_DEBUG = os.getenv("API_DEBUG", "false").lower() == "true"

        # CORS Configuration
        cors_origins = os.getenv("CORS_ORIGINS", "*")
        self.CORS_ORIGINS = cors_origins.split(",") if cors_origins != "*" else ["*"]

        # SaaS authentication and local credential-vault configuration.
        self.JWT_SECRET = os.getenv("VALUECELL_JWT_SECRET", "development-only-change-me")
        self.JWT_ISSUER = os.getenv("VALUECELL_JWT_ISSUER", "valuecell-saas")
        self.JWT_ACCESS_TOKEN_TTL_S = int(
            os.getenv("VALUECELL_JWT_ACCESS_TOKEN_TTL_S", "3600")
        )
        self.CREDENTIAL_MASTER_KEY = os.getenv("VALUECELL_CREDENTIAL_MASTER_KEY")
        if (
            self.APP_ENVIRONMENT.lower() in {"production", "prod"}
            and self.JWT_SECRET == "development-only-change-me"
        ):
            raise RuntimeError("VALUECELL_JWT_SECRET must be configured in production")

        # Live execution remains impossible unless explicitly enabled at runtime.
        self.LIVE_TRADING_ENABLED = (
            os.getenv("VALUECELL_LIVE_TRADING_ENABLED", "false").lower() == "true"
        )
        self.LIVE_AUTHORIZATION_TTL_S = _positive_int_env(
            "VALUECELL_LIVE_AUTHORIZATION_TTL_S", 900
        )

        # Database Configuration
        # Prefer `VALUECELL_DATABASE_URL` if provided; otherwise use system application directory default.
        env_db = os.getenv("VALUECELL_DATABASE_URL")
        if env_db:
            # If it's already a full DSN (sqlite or other), use as-is
            self.DATABASE_URL = env_db
        else:
            self.DATABASE_URL = _default_db_path()

        # File Paths
        self.BASE_DIR = PROJECT_ROOT
        self.LOGS_DIR = self.BASE_DIR / "logs"
        self.LOGS_DIR.mkdir(exist_ok=True)

        # I18n Configuration
        self.LOCALE_DIR = self.BASE_DIR / "configs/locales"

        # Public market-data reliability configuration. These providers require no API keys.
        self.MARKET_DATA_PROVIDERS = _parse_market_data_providers(
            os.getenv("VALUECELL_MARKET_DATA_PROVIDERS", "okx,binance,gate,mexc")
        )
        self.MARKET_DEFAULT_SYMBOLS = _parse_default_market_symbols(
            os.getenv("VALUECELL_MARKET_DEFAULT_SYMBOLS", "BTC-USDT,ETH-USDT,SOL-USDT")
        )
        self.MARKET_DEFAULT_INTERVAL = _default_market_interval(
            os.getenv("VALUECELL_MARKET_DEFAULT_INTERVAL", "1h")
        )
        self.MARKET_DEFAULT_LOOKBACK = _positive_int_env(
            "VALUECELL_MARKET_DEFAULT_LOOKBACK", 240
        )
        if self.MARKET_DEFAULT_LOOKBACK > 500:
            raise ValueError("VALUECELL_MARKET_DEFAULT_LOOKBACK must be at most 500")
        self.MARKET_REFRESH_S = _positive_int_env("VALUECELL_MARKET_REFRESH_S", 3600)
        if self.MARKET_REFRESH_S < 60:
            raise ValueError("VALUECELL_MARKET_REFRESH_S must be at least 60")
        self.MARKET_DATA_MAX_CONCURRENT_FETCHES = _positive_int_env(
            "VALUECELL_MARKET_DATA_MAX_CONCURRENT_FETCHES", 3
        )
        self.MARKET_DATA_CACHE_TTL_S = _positive_float_env(
            "VALUECELL_MARKET_DATA_CACHE_TTL_S", 30.0
        )
        self.MARKET_DATA_FAILURE_COOLDOWN_BASE_S = _positive_float_env(
            "VALUECELL_MARKET_DATA_FAILURE_COOLDOWN_BASE_S", 30.0
        )
        self.MARKET_DATA_FAILURE_COOLDOWN_MAX_S = _positive_float_env(
            "VALUECELL_MARKET_DATA_FAILURE_COOLDOWN_MAX_S", 300.0
        )
        if (
            self.MARKET_DATA_FAILURE_COOLDOWN_MAX_S
            < self.MARKET_DATA_FAILURE_COOLDOWN_BASE_S
        ):
            raise ValueError(
                "VALUECELL_MARKET_DATA_FAILURE_COOLDOWN_MAX_S must be at least "
                "VALUECELL_MARKET_DATA_FAILURE_COOLDOWN_BASE_S"
            )



    def get_database_config(self) -> dict:
        """Get database configuration."""
        return {"url": self.DATABASE_URL}

    def update_language(self, language: str) -> None:
        """Update current language setting.

        Args:
            language: Language code to set
        """
        # In a production environment, this might update a database or config file
        # For now, we'll just log the change
        pass

    def update_timezone(self, timezone: str) -> None:
        """Update current timezone setting.

        Args:
            timezone: Timezone to set
        """
        # In a production environment, this might update a database or config file
        # For now, we'll just log the change
        pass


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
