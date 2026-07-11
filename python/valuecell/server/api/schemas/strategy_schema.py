"""Dynamic strategy configuration schema contracts."""

from typing import Any, List, Literal, Optional

from pydantic import BaseModel, Field

from .base import SuccessResponse


class StrategyConfigOption(BaseModel):
    label: str
    value: Any


class StrategyConfigField(BaseModel):
    key: str
    label: str
    field_type: Literal["text", "number", "boolean", "select", "multi_select", "number_list"]
    default: Any = None
    description: Optional[str] = None
    min: Optional[float] = None
    max: Optional[float] = None
    step: Optional[float] = None
    options: List[StrategyConfigOption] = Field(default_factory=list)
    required: bool = False
    group: str = "general"
    persistence_target: Literal["trading_config", "strategy_params"] = "strategy_params"


class StrategyConfigSchema(BaseModel):
    strategy_type: str
    label: str
    description: str
    defaults: dict[str, Any]
    fields: List[StrategyConfigField]


class StrategyConfigSchemaCatalog(BaseModel):
    schemas: List[StrategyConfigSchema]


StrategyConfigSchemaCatalogResponse = SuccessResponse[StrategyConfigSchemaCatalog]
