import type { TFunction } from "i18next";
import { z } from "zod";

export const createAiModelSchema = (t: TFunction) =>
  z.object({
    provider: z.string(),
    model_id: z.string(),
    api_key: z.string(),
  });

const ruleBasedStrategyTypes = [
  "LongTermSpotRsiStrategy",
  "ShortTermSpotRsiStrategy",
  "ColdAltSpotStrategy",
] as const;

const strategyTypes = [
  "PromptBasedStrategy",
  "GridStrategy",
  ...ruleBasedStrategyTypes,
] as const;

const isRuleBasedStrategyType = (
  value: (typeof strategyTypes)[number],
): value is (typeof ruleBasedStrategyTypes)[number] =>
  ruleBasedStrategyTypes.includes(value as (typeof ruleBasedStrategyTypes)[number]);

const baseStep2Fields = {
  exchange_id: z.string(),
  api_key: z.string(),
  secret_key: z.string(),
  passphrase: z.string(),
  wallet_address: z.string(),
  private_key: z.string(),
};

// Step 2 Schema: Exchanges (conditional validation with superRefine)
export const createExchangeSchema = (t: TFunction) =>
  z.union([
    z.object({
      ...baseStep2Fields,
      trading_mode: z.literal("virtual"),
    }),

    // Live Trading - Hyperliquid
    z.object({
      ...baseStep2Fields,
      trading_mode: z.literal("live"),
      exchange_id: z.literal("hyperliquid"),
      wallet_address: z
        .string()
        .min(1, t("validation.exchange.walletAddressHyperliquidRequired")),
      private_key: z
        .string()
        .min(1, t("validation.exchange.privateKeyHyperliquidRequired")),
    }),

    // Live Trading - OKX & Coinbase (Require Passphrase)
    z.object({
      ...baseStep2Fields,
      trading_mode: z.literal("live"),
      exchange_id: z.enum(["okx", "coinbaseexchange"]),
      api_key: z.string().min(1, t("validation.exchange.apiKeyRequired")),
      secret_key: z.string().min(1, t("validation.exchange.secretKeyRequired")),
      passphrase: z
        .string()
        .min(1, t("validation.exchange.passphraseRequired")),
    }),

    // Live Trading - Standard Exchanges
    z.object({
      ...baseStep2Fields,
      trading_mode: z.literal("live"),
      exchange_id: z.enum(["binance", "blockchaincom", "gate", "mexc"]),
      api_key: z.string().min(1, t("validation.exchange.apiKeyRequired")),
      secret_key: z.string().min(1, t("validation.exchange.secretKeyRequired")),
    }),
  ]);

// Step 3 Schema: Trading Strategy
export const createTradingStrategySchema = (t: TFunction) =>
  z.object({
    strategy_type: z.enum(strategyTypes),
    strategy_name: z
      .string()
      .min(1, t("validation.trading.strategyNameRequired")),
    initial_capital: z
      .number()
      .min(1, t("validation.trading.initialCapitalMin")),
    max_leverage: z
      .number()
      .min(1, t("validation.trading.maxLeverageMin"))
      .max(5, t("validation.trading.maxLeverageMax")),
    symbols: z.array(z.string()).min(1, t("validation.trading.symbolsMin")),
    template_id: z.string(),
    decide_interval: z
      .number()
      .min(10, t("validation.trading.decideIntervalMin"))
      .max(3600, t("validation.trading.decideIntervalMax")),
  })
  .superRefine((value, ctx) => {
    if (
      value.strategy_type === "PromptBasedStrategy" &&
      (!value.template_id || value.template_id.trim().length === 0)
    ) {
      ctx.addIssue({
        code: z.ZodIssueCode.custom,
        path: ["template_id"],
        message: t("validation.trading.templateRequired"),
      });
    }

    if (isRuleBasedStrategyType(value.strategy_type) && value.max_leverage !== 1) {
      ctx.addIssue({
        code: z.ZodIssueCode.custom,
        path: ["max_leverage"],
        message: t("validation.trading.maxLeverageSpotOnly"),
      });
    }
  });

export const createCopyTradingStrategySchema = (t: TFunction) =>
  z.object({
    strategy_name: z
      .string()
      .min(1, t("validation.trading.strategyNameRequired")),
    initial_capital: z
      .number()
      .min(1, t("validation.trading.initialCapitalMin")),
    max_leverage: z
      .number()
      .min(1, t("validation.trading.maxLeverageMin"))
      .max(5, t("validation.trading.maxLeverageMax")),
    symbols: z.array(z.string()).min(1, t("validation.trading.symbolsMin")),
    decide_interval: z
      .number()
      .min(10, t("validation.trading.decideIntervalMin"))
      .max(3600, t("validation.trading.decideIntervalMax")),
    strategy_type: z.enum(strategyTypes),
    prompt_name: z.string(),
    prompt: z.string(),
  })
  .superRefine((value, ctx) => {
    if (value.strategy_type === "PromptBasedStrategy") {
      if (!value.prompt_name || value.prompt_name.trim().length === 0) {
        ctx.addIssue({
          code: z.ZodIssueCode.custom,
          path: ["prompt_name"],
          message: t("validation.copy.promptNameRequired"),
        });
      }
      if (!value.prompt || value.prompt.trim().length === 0) {
        ctx.addIssue({
          code: z.ZodIssueCode.custom,
          path: ["prompt"],
          message: t("validation.copy.promptRequired"),
        });
      }
    }

    if (isRuleBasedStrategyType(value.strategy_type) && value.max_leverage !== 1) {
      ctx.addIssue({
        code: z.ZodIssueCode.custom,
        path: ["max_leverage"],
        message: t("validation.trading.maxLeverageSpotOnly"),
      });
    }
  });
