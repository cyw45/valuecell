import {
  createContext,
  type PropsWithChildren,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import { Platform } from "react-native";
import * as SecureStore from "expo-secure-store";

export type MobileLanguage = "en" | "zh_CN" | "zh_TW" | "ja";
export type MobileThemePreference = "dark" | "light" | "system";
export type StockColorMode = "GREEN_UP_RED_DOWN" | "RED_UP_GREEN_DOWN";
export type MarketDataRefreshMode = "manual" | "5s" | "15s" | "30s" | "1m" | "5m";

export interface MobilePreferences {
  language: MobileLanguage;
  theme: MobileThemePreference;
  stockColorMode: StockColorMode;
  marketDataRefreshMode: MarketDataRefreshMode;
}

export const MOBILE_PREFERENCES_KEY = "valuecell.mobile.preferences";

export const DEFAULT_MOBILE_PREFERENCES: MobilePreferences = {
  language: "zh_CN",
  theme: "dark",
  stockColorMode: "GREEN_UP_RED_DOWN",
  marketDataRefreshMode: "15s",
};

export const MARKET_DATA_REFRESH_INTERVAL_MS: Readonly<Record<MarketDataRefreshMode, false | number>> = {
  manual: false,
  "5s": 5_000,
  "15s": 15_000,
  "30s": 30_000,
  "1m": 60_000,
  "5m": 300_000,
};

export function marketDataRefreshInterval(mode: MarketDataRefreshMode): false | number {
  return MARKET_DATA_REFRESH_INTERVAL_MS[mode];
}

interface WebStorage {
  getItem(key: string): string | null;
  setItem(key: string, value: string): void;
  removeItem(key: string): void;
}

function webStorage(): WebStorage | null {
  if (Platform.OS !== "web") return null;
  try {
    return (globalThis as unknown as { localStorage?: WebStorage }).localStorage ?? null;
  } catch {
    return null;
  }
}

async function readStoredPreferences(): Promise<string | null> {
  if (Platform.OS === "web") return webStorage()?.getItem(MOBILE_PREFERENCES_KEY) ?? null;
  return SecureStore.getItemAsync(MOBILE_PREFERENCES_KEY);
}

async function saveStoredPreferences(value: string): Promise<void> {
  if (Platform.OS === "web") {
    webStorage()?.setItem(MOBILE_PREFERENCES_KEY, value);
    return;
  }
  await SecureStore.setItemAsync(MOBILE_PREFERENCES_KEY, value);
}

async function removeStoredPreferences(): Promise<void> {
  if (Platform.OS === "web") {
    webStorage()?.removeItem(MOBILE_PREFERENCES_KEY);
    return;
  }
  await SecureStore.deleteItemAsync(MOBILE_PREFERENCES_KEY);
}

function valueIn<T extends string>(value: unknown, allowed: readonly T[], fallback: T): T {
  return typeof value === "string" && (allowed as readonly string[]).includes(value)
    ? value as T
    : fallback;
}

function normalizePreferences(value: unknown): MobilePreferences {
  if (!value || typeof value !== "object") return DEFAULT_MOBILE_PREFERENCES;
  const candidate = value as Partial<MobilePreferences>;
  return {
    language: valueIn(candidate.language, ["en", "zh_CN", "zh_TW", "ja"], DEFAULT_MOBILE_PREFERENCES.language),
    theme: valueIn(candidate.theme, ["dark", "light", "system"], DEFAULT_MOBILE_PREFERENCES.theme),
    stockColorMode: valueIn(candidate.stockColorMode, ["GREEN_UP_RED_DOWN", "RED_UP_GREEN_DOWN"], DEFAULT_MOBILE_PREFERENCES.stockColorMode),
    marketDataRefreshMode: valueIn(candidate.marketDataRefreshMode, ["manual", "5s", "15s", "30s", "1m", "5m"], DEFAULT_MOBILE_PREFERENCES.marketDataRefreshMode),
  };
}

export async function loadMobilePreferences(): Promise<MobilePreferences> {
  const stored = await readStoredPreferences();
  if (!stored) return DEFAULT_MOBILE_PREFERENCES;
  try {
    return normalizePreferences(JSON.parse(stored));
  } catch {
    return DEFAULT_MOBILE_PREFERENCES;
  }
}

export async function persistMobilePreferences(preferences: MobilePreferences): Promise<void> {
  await saveStoredPreferences(JSON.stringify(normalizePreferences(preferences)));
}

export async function clearMobilePreferences(): Promise<void> {
  await removeStoredPreferences();
}

type PreferencesContextValue = {
  preferences: MobilePreferences;
  ready: boolean;
  updatePreferences: (changes: Partial<MobilePreferences>) => Promise<MobilePreferences>;
  replacePreferences: (next: MobilePreferences) => Promise<MobilePreferences>;
  setLanguage: (language: MobileLanguage) => Promise<MobilePreferences>;
  setTheme: (theme: MobileThemePreference) => Promise<MobilePreferences>;
  setStockColorMode: (stockColorMode: StockColorMode) => Promise<MobilePreferences>;
  setMarketDataRefreshMode: (marketDataRefreshMode: MarketDataRefreshMode) => Promise<MobilePreferences>;
};

const PreferencesContext = createContext<PreferencesContextValue | null>(null);

export function PreferencesProvider({ children }: PropsWithChildren) {
  const [preferences, setPreferences] = useState<MobilePreferences>(DEFAULT_MOBILE_PREFERENCES);
  const [ready, setReady] = useState(false);
  const preferencesRef = useRef(preferences);

  useEffect(() => {
    let mounted = true;
    void loadMobilePreferences()
      .then((next) => {
        if (!mounted) return;
        preferencesRef.current = next;
        setPreferences(next);
      })
      .catch(() => {
        if (!mounted) return;
        preferencesRef.current = DEFAULT_MOBILE_PREFERENCES;
        setPreferences(DEFAULT_MOBILE_PREFERENCES);
      })
      .finally(() => {
        if (mounted) setReady(true);
      });
    return () => {
      mounted = false;
    };
  }, []);

  const replacePreferences = useCallback(async (next: MobilePreferences) => {
    const normalized = normalizePreferences(next);
    const previous = preferencesRef.current;
    preferencesRef.current = normalized;
    setPreferences(normalized);
    try {
      await persistMobilePreferences(normalized);
      return normalized;
    } catch (reason) {
      if (preferencesRef.current === normalized) {
        preferencesRef.current = previous;
        setPreferences(previous);
      }
      throw reason;
    }
  }, []);

  const updatePreferences = useCallback(
    (changes: Partial<MobilePreferences>) => replacePreferences({ ...preferencesRef.current, ...changes }),
    [replacePreferences],
  );

  const value = useMemo<PreferencesContextValue>(() => ({
    preferences,
    ready,
    updatePreferences,
    replacePreferences,
    setLanguage: (language) => updatePreferences({ language }),
    setTheme: (theme) => updatePreferences({ theme }),
    setStockColorMode: (stockColorMode) => updatePreferences({ stockColorMode }),
    setMarketDataRefreshMode: (marketDataRefreshMode) => updatePreferences({ marketDataRefreshMode }),
  }), [preferences, ready, replacePreferences, updatePreferences]);

  return <PreferencesContext.Provider value={value}>{children}</PreferencesContext.Provider>;
}

export function usePreferences(): PreferencesContextValue {
  const value = useContext(PreferencesContext);
  if (!value) throw new Error("usePreferences must be used inside PreferencesProvider");
  return value;
}
