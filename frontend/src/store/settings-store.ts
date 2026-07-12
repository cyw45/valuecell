import { create } from "zustand";
import { devtools, persist } from "zustand/middleware";
import { useShallow } from "zustand/shallow";
import {
  GREEN_BADGE,
  GREEN_COLOR,
  GREEN_GRADIENT,
  NEUTRAL_BADGE,
  NEUTRAL_COLOR,
  NEUTRAL_GRADIENT,
  RED_BADGE,
  RED_COLOR,
  RED_GRADIENT,
} from "@/constants/stock";
import i18n from "@/i18n";
import type { StockChangeType } from "@/types/stock";

export type StockColorMode = "GREEN_UP_RED_DOWN" | "RED_UP_GREEN_DOWN";
export type LanguageCode = "en" | "zh_CN" | "zh_TW" | "ja";
export type MarketDataRefreshMode = "manual" | "5s" | "15s" | "30s" | "1m" | "5m";
export const DEFAULT_LANGUAGE = "zh_CN";
export const DEFAULT_MARKET_DATA_REFRESH_MODE: MarketDataRefreshMode = "15s";

export const MARKET_DATA_REFRESH_INTERVAL_MS: Record<MarketDataRefreshMode, false | number> = {
  manual: false,
  "5s": 5_000,
  "15s": 15_000,
  "30s": 30_000,
  "1m": 60_000,
  "5m": 300_000,
};

export const getMarketDataRefreshIntervalMs = (
  mode: MarketDataRefreshMode,
  strategyIntervalSeconds?: number,
) => {
  if (strategyIntervalSeconds && strategyIntervalSeconds > 0) {
    return Math.max(strategyIntervalSeconds * 1000, 5_000);
  }
  return MARKET_DATA_REFRESH_INTERVAL_MS[mode];
};

interface SettingsStoreState {
  stockColorMode: StockColorMode;
  language: LanguageCode;
  marketDataRefreshMode: MarketDataRefreshMode;
  setStockColorMode: (mode: StockColorMode) => void;
  setLanguage: (language: LanguageCode) => void;
  setMarketDataRefreshMode: (mode: MarketDataRefreshMode) => void;
}

const getLanguage = () => {
  if (typeof navigator === "undefined") {
    return DEFAULT_LANGUAGE;
  }
  const map: Record<string, LanguageCode> = {
    "zh-CN": "zh_CN",
    "zh-Hans": "zh_CN",
    "zh-TW": "zh_TW",
    "zh-Hant": "zh_TW",
    "ja-JP": "ja",
  };
  return map[navigator.language] ?? DEFAULT_LANGUAGE;
};

const INITIAL_STATE = {
  stockColorMode: "GREEN_UP_RED_DOWN" as StockColorMode,
  language: getLanguage() as LanguageCode,
  marketDataRefreshMode: DEFAULT_MARKET_DATA_REFRESH_MODE,
};

/**
 * Global settings store with localStorage persistence
 */
export const useSettingsStore = create<SettingsStoreState>()(
  devtools(
    persist(
      (set) => ({
        ...INITIAL_STATE,
        setStockColorMode: (stockColorMode) => set({ stockColorMode }),
        setMarketDataRefreshMode: (marketDataRefreshMode) =>
          set({ marketDataRefreshMode }),
        setLanguage: (language) => {
          set({ language });
          i18n.changeLanguage(language);
        },
      }),
      {
        name: "valuecell-settings",
      },
    ),
    { name: "SettingsStore", enabled: import.meta.env.DEV },
  ),
);

export const useStockColorMode = () =>
  useSettingsStore(useShallow((s) => s.stockColorMode));

export const useLanguage = () =>
  useSettingsStore(useShallow((s) => s.language));

export const useMarketDataRefreshMode = () =>
  useSettingsStore(useShallow((s) => s.marketDataRefreshMode));

export const useSettingsActions = () =>
  useSettingsStore(
    useShallow((s) => ({
      setStockColorMode: s.setStockColorMode,
      setLanguage: s.setLanguage,
      setMarketDataRefreshMode: s.setMarketDataRefreshMode,
    })),
  );

/**
 * Get stock colors based on current color mode setting
 */
export const useStockColors = (): Record<StockChangeType, string> => {
  const colorMode = useStockColorMode();
  if (colorMode === "RED_UP_GREEN_DOWN") {
    return {
      positive: RED_COLOR,
      negative: GREEN_COLOR,
      neutral: NEUTRAL_COLOR,
    };
  }
  return {
    positive: GREEN_COLOR,
    negative: RED_COLOR,
    neutral: NEUTRAL_COLOR,
  };
};

/**
 * Get stock gradient colors based on current color mode setting
 */
export const useStockGradientColors = (): Record<
  StockChangeType,
  [string, string]
> => {
  const colorMode = useStockColorMode();
  if (colorMode === "RED_UP_GREEN_DOWN") {
    return {
      positive: RED_GRADIENT,
      negative: GREEN_GRADIENT,
      neutral: NEUTRAL_GRADIENT,
    };
  }
  return {
    positive: GREEN_GRADIENT,
    negative: RED_GRADIENT,
    neutral: NEUTRAL_GRADIENT,
  };
};

/**
 * Get stock badge colors based on current color mode setting
 */
export const useStockBadgeColors = (): Record<
  StockChangeType,
  { bg: string; text: string }
> => {
  const colorMode = useStockColorMode();
  if (colorMode === "RED_UP_GREEN_DOWN") {
    return {
      positive: RED_BADGE,
      negative: GREEN_BADGE,
      neutral: NEUTRAL_BADGE,
    };
  }
  return {
    positive: GREEN_BADGE,
    negative: RED_BADGE,
    neutral: NEUTRAL_BADGE,
  };
};
