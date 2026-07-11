import json

from fastapi import FastAPI
from fastapi.testclient import TestClient

from valuecell.server.api.routers.strategy_schema import create_strategy_schema_router
from valuecell.server.services.strategy_schema_service import StrategySchemaService


EXPECTED_STRATEGY_TYPES = {
    "PromptBasedStrategy",
    "GridStrategy",
    "LongTermSpotRsiStrategy",
    "ShortTermSpotRsiStrategy",
}

PERSISTED_TRADING_FIELDS = {
    "symbols",
    "decide_interval",
    "max_leverage",
    "initial_capital",
    "max_positions",
    "cap_factor",
}

RSI_STRATEGY_FIELDS = {
    "primary_interval",
    "entry_rsi_thresholds",
    "sell_rsi_thresholds",
    "daily_overbought_rsi",
    "bear_cap_ratio",
    "max_additions",
}


def _dump(value):
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, dict):
        return value
    raise AssertionError(f"Unexpected schema payload type: {type(value)!r}")


def _catalog_items(payload):
    data = _dump(payload)
    for key in ("schemas", "strategies", "items"):
        items = data.get(key)
        if isinstance(items, list):
            return items
    if isinstance(data, list):
        return data
    raise AssertionError(f"Could not find schema item list in payload keys: {sorted(data)}")


def _strategy_type(item):
    return item.get("strategy_type") or item.get("type") or item.get("id")


def _properties(item):
    schema = item.get("schema") or item.get("json_schema") or item.get("parameters") or item
    properties = schema.get("properties") if isinstance(schema, dict) else None
    if isinstance(properties, dict):
        return properties
    fields = item.get("fields")
    if isinstance(fields, list):
        return {
            field.get("key") or field.get("name"): field
            for field in fields
            if isinstance(field, dict)
        }
    if isinstance(fields, dict):
        return fields
    raise AssertionError(f"Could not find field definitions for {_strategy_type(item)}")


def _fields_by_key(item):
    fields = item.get("fields")
    assert isinstance(fields, list), f"Fields must be a list for {_strategy_type(item)}"
    return {field["key"]: field for field in fields}


def _defaults(item):
    defaults = item.get("defaults") or item.get("default_values")
    if defaults is None:
        schema = item.get("schema") or item.get("json_schema") or {}
        defaults = {
            name: field["default"]
            for name, field in (schema.get("properties") or {}).items()
            if isinstance(field, dict) and "default" in field
        }
    assert isinstance(defaults, dict), f"Defaults must be a dict for {_strategy_type(item)}"
    return defaults


def _assert_has_any_field(properties, candidates, strategy_type):
    assert any(name in properties for name in candidates), (
        f"{strategy_type} schema missing one of {sorted(candidates)}; "
        f"available fields: {sorted(properties)}"
    )


def test_strategy_schema_service_catalog_includes_all_supported_strategy_types():
    catalog = StrategySchemaService().get_catalog()
    items = _catalog_items(catalog)
    by_type = {_strategy_type(item): item for item in items}

    assert EXPECTED_STRATEGY_TYPES <= set(by_type)


def test_strategy_schema_service_declares_configurable_fields_and_persistence_targets():
    catalog = StrategySchemaService().get_catalog()
    by_type = {_strategy_type(item): item for item in _catalog_items(catalog)}

    for strategy_type in EXPECTED_STRATEGY_TYPES:
        item = by_type[strategy_type]
        fields = _fields_by_key(item)
        defaults = _defaults(item)

        assert PERSISTED_TRADING_FIELDS <= set(fields), strategy_type
        assert {
            field_name: fields[field_name]["persistence_target"]
            for field_name in PERSISTED_TRADING_FIELDS
        } == {
            field_name: "trading_config" for field_name in PERSISTED_TRADING_FIELDS
        }
        assert set(defaults["symbols"]) <= {
            option["value"] for option in fields["symbols"]["options"]
        }
        assert fields["symbols"]["field_type"] == "multi_select"
        assert fields["decide_interval"]["field_type"] == "number"
        assert fields["decide_interval"]["min"] == 10
        assert fields["decide_interval"]["max"] == 3600
        assert fields["decide_interval"]["step"] == 5
        json.dumps(defaults)

    prompt_fields = _fields_by_key(by_type["PromptBasedStrategy"])
    assert prompt_fields["template_id"]["persistence_target"] == "trading_config"

    for strategy_type in ("LongTermSpotRsiStrategy", "ShortTermSpotRsiStrategy"):
        fields = _fields_by_key(by_type[strategy_type])
        assert RSI_STRATEGY_FIELDS <= set(fields), strategy_type
        assert {
            field_name: fields[field_name]["persistence_target"]
            for field_name in RSI_STRATEGY_FIELDS
        } == {
            field_name: "strategy_params" for field_name in RSI_STRATEGY_FIELDS
        }
        assert fields["entry_rsi_thresholds"]["field_type"] == "number_list"
        assert fields["sell_rsi_thresholds"]["field_type"] == "number_list"
        assert fields["daily_overbought_rsi"]["min"] == 1
        assert fields["daily_overbought_rsi"]["max"] == 100


def test_strategy_schema_endpoint_returns_success_response():
    app = FastAPI()
    app.include_router(create_strategy_schema_router())

    response = TestClient(app).get("/strategies/schemas")

    assert response.status_code == 200
    body = response.json()
    assert body["code"] == 0
    items = _catalog_items(body["data"])
    assert EXPECTED_STRATEGY_TYPES <= {_strategy_type(item) for item in items}
