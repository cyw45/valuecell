"""
ValueCell Server - Database Models

This package contains all database models for the ValueCell server.
All models are automatically imported to ensure they are registered with SQLAlchemy.
"""

# Import all models to ensure they are registered with SQLAlchemy
from .agent import Agent
from .asset import Asset

# Import base model
from .base import Base
from .strategy import Strategy
from .strategy_compose_cycle import StrategyComposeCycle
from .strategy_detail import StrategyDetail
from .strategy_cycle_diagnostics import StrategyCycleDiagnostics
from .strategy_holding import StrategyHolding
from .strategy_instruction import StrategyInstruction
from .strategy_portfolio import StrategyPortfolioView
from .rule_strategy import (
    RuleStrategy,
    RuleStrategyAccount,
    RuleStrategyDemoAccountSnapshot,
    RuleStrategyEvaluationJournal,
    RuleStrategyEvent,
    RuleStrategyExecutionIntent,
    RuleStrategyExecutionLease,
    RuleStrategyFill,
    RuleStrategyMonitorSymbol,
    RuleStrategyOrderAttempt,
    RuleStrategyRiskState,
)
from .rule_strategy_validation import (
    RuleStrategyValidationDataset,
    RuleStrategyValidationFill,
    RuleStrategyValidationPoint,
    RuleStrategyValidationRun,
)
from .rule_strategy_manual_close import RuleStrategyManualCloseCommand
from .rule_strategy_text_import_job import RuleStrategyTextImportJobRecord
from .tenant import SaaSUser, Tenant, TenantMembership, TenantProfile
from .tenant_credential import TenantCredential
from .sandbox_exchange_order import SandboxExchangeOrder
from .live_execution import LiveExecutionOrder, LiveRiskPolicy, LiveStrategyBinding
from .saas_control import (
    AuditEvent,
    EnterpriseAgreement,
    ProfitSettlement,
    ServicePlan,
    TenantSubscription,
)
from .user_profile import ProfileCategory, UserProfile
from .watchlist import Watchlist, WatchlistItem
from .world_intelligence import WorldIntelligenceSnapshot

# Export all models
__all__ = [
    "Base",
    "Agent",
    "Asset",
    "Strategy",
    "Watchlist",
    "WatchlistItem",
    "UserProfile",
    "ProfileCategory",
    "StrategyHolding",
    "StrategyDetail",
    "StrategyCycleDiagnostics",
    "StrategyPortfolioView",
    "StrategyComposeCycle",
    "StrategyInstruction",
    "RuleStrategy",
    "RuleStrategyAccount",
    "RuleStrategyEvaluationJournal",
    "RuleStrategyDemoAccountSnapshot",
    "RuleStrategyEvent",
    "RuleStrategyExecutionIntent",
    "RuleStrategyExecutionLease",
    "RuleStrategyFill",
    "RuleStrategyMonitorSymbol",
    "RuleStrategyOrderAttempt",
    "RuleStrategyRiskState",
    "RuleStrategyValidationDataset",
    "RuleStrategyValidationFill",
    "RuleStrategyManualCloseCommand",
    "RuleStrategyValidationPoint",
    "RuleStrategyValidationRun",
    "RuleStrategyTextImportJobRecord",
    "SaaSUser",
    "Tenant",
    "TenantMembership",
    "TenantProfile",
    "TenantCredential",
    "SandboxExchangeOrder",
    "LiveExecutionOrder",
    "LiveRiskPolicy",
    "LiveStrategyBinding",
    "AuditEvent",
    "EnterpriseAgreement",
    "ProfitSettlement",
    "ServicePlan",
    "TenantSubscription",
    "WorldIntelligenceSnapshot",
]
