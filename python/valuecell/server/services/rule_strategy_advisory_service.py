"""Read-only AI advisory analysis for paper rule-strategy configurations."""

from __future__ import annotations

import json
from typing import Any

import requests
from loguru import logger



class RuleStrategyAdvisoryUnavailableError(Exception):
    """Raised when no configured advisory model can safely be used."""


class RuleStrategyAdvisoryService:
    """Generate non-executable strategy configuration reviews.

    This service is intentionally separate from RuleEngine, scheduler, execution,
    and persistence. Its output cannot mutate a strategy or create an order.
    """

    def review_configuration(
        self, strategy: dict[str, Any], latest_evaluations: list[dict[str, Any]]
    ) -> dict[str, Any]:
        from valuecell.config.manager import get_config_manager

        manager = get_config_manager()
        provider = manager.primary_provider
        provider_config = manager.get_provider_config(provider)
        if provider != "openai-compatible":
            raise RuleStrategyAdvisoryUnavailableError(
                "AI advisory requires the configured HiCode OpenAI-compatible provider"
            )
        if provider_config is None or not provider_config.api_key or not provider_config.base_url:
            raise RuleStrategyAdvisoryUnavailableError(
                "AI advisory is unavailable because the HiCode proxy is not configured"
            )
        model_id = provider_config.default_model
        if not model_id:
            raise RuleStrategyAdvisoryUnavailableError(
                "AI advisory is unavailable because the HiCode provider has no default model"
            )

        prompt = self._build_prompt(strategy, latest_evaluations)
        try:
            response = requests.post(
                f"{provider_config.base_url.rstrip('/')}/chat/completions",
                headers={"Authorization": f"Bearer {provider_config.api_key}"},
                json={
                    "model": model_id,
                    "temperature": 0.2,
                    "max_tokens": 900,
                    "messages": [
                        {
                            "role": "system",
                            "content": (
                                "You are a read-only paper-trading strategy reviewer. "
                                "Never recommend a trade or an order. Never claim performance "
                                "not present in the evidence. Give concise configuration and risk "
                                "observations with evidence. Recommendations cannot imply automatic changes."
                            ),
                        },
                        {"role": "user", "content": prompt},
                    ],
                },
                timeout=30,
            )
            response.raise_for_status()
            payload = response.json()
            content = str(payload["choices"][0]["message"]["content"]).strip()
        except (KeyError, IndexError, TypeError, ValueError, requests.RequestException) as exc:
            logger.warning("Rule strategy advisory HiCode request failed: {}", exc)
            raise RuleStrategyAdvisoryUnavailableError(
                "AI advisory could not be generated from the configured HiCode proxy"
            ) from exc

        if not content:
            raise RuleStrategyAdvisoryUnavailableError("AI advisory returned no content")
        return {
            "kind": "configuration_review",
            "authority": "advisory_only",
            "provider": provider,
            "model_id": model_id,
            "content": content,
        }

    @staticmethod
    def _build_prompt(
        strategy: dict[str, Any], latest_evaluations: list[dict[str, Any]]
    ) -> str:
        evidence = {
            "strategy_id": strategy["strategy_id"],
            "name": strategy["name"],
            "mode": strategy["mode"],
            "config": strategy["config"],
            "latest_evaluations": latest_evaluations[:10],
        }
        return (
            "Review this saved paper-rule strategy configuration and its most recent "
            "deterministic evaluation records. Identify only observable configuration, "
            "risk, data-sufficiency, or condition-frequency concerns. Do not change it.\n\n"
            f"Evidence:\n{json.dumps(evidence, ensure_ascii=False, default=str)}"
        )
