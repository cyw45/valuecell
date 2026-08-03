import { createContext, type PropsWithChildren, useContext, useMemo } from "react";
import { useColorScheme } from "react-native";
import { usePreferences } from "./preferences";
import {
  palette,
  quoteColorsFor,
  resolveTheme,
  resolveThemeTokens,
  type QuoteColors,
  type ResolvedTheme,
  type ThemeTokens,
} from "./theme";

type ThemeContextValue = {
  tokens: ThemeTokens;
  quoteColors: QuoteColors;
  resolvedTheme: ResolvedTheme;
  isDark: boolean;
};

const ThemeContext = createContext<ThemeContextValue | null>(null);

export function ThemeProvider({ children }: PropsWithChildren) {
  const { preferences } = usePreferences();
  const systemColorScheme = useColorScheme();
  const resolvedTheme = resolveTheme(preferences.theme, systemColorScheme);
  const value = useMemo<ThemeContextValue>(() => {
    const tokens = resolveThemeTokens(preferences.theme, systemColorScheme);
    return {
      tokens,
      quoteColors: quoteColorsFor(preferences.stockColorMode, tokens),
      resolvedTheme,
      isDark: resolvedTheme === "dark",
    };
  }, [preferences.stockColorMode, preferences.theme, resolvedTheme, systemColorScheme]);
  Object.assign(palette, value.tokens);

  return <ThemeContext.Provider key={`${value.resolvedTheme}:${preferences.stockColorMode}`} value={value}>{children}</ThemeContext.Provider>;
}

export function useTheme(): ThemeContextValue {
  const value = useContext(ThemeContext);
  if (!value) throw new Error("useTheme must be used inside ThemeProvider");
  return value;
}
