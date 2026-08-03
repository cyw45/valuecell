import type { ColorSchemeName } from "react-native";
import type { MobileThemePreference, StockColorMode } from "./preferences";

export type ThemePalette = Readonly<{
  canvas: string;
  surface: string;
  surfaceRaised: string;
  surfaceMuted: string;
  border: string;
  text: string;
  textMuted: string;
  primary: string;
  primarySoft: string;
  positive: string;
  positiveSoft: string;
  negative: string;
  negativeSoft: string;
  warning: string;
  warningSoft: string;
  white: string;
}>;

export const darkPalette: ThemePalette = {
  canvas: "#07111F",
  surface: "#0D1A2B",
  surfaceRaised: "#12253A",
  surfaceMuted: "#172B41",
  border: "#24435C",
  text: "#F5FAFF",
  textMuted: "#91A9BC",
  primary: "#2AB5F6",
  primarySoft: "#163C59",
  positive: "#27D6A3",
  positiveSoft: "#123F3A",
  negative: "#FB7185",
  negativeSoft: "#4A2030",
  warning: "#F5B544",
  warningSoft: "#4B3514",
  white: "#FFFFFF",
};

export const lightPalette: ThemePalette = {
  canvas: "#F4F8FC",
  surface: "#FFFFFF",
  surfaceRaised: "#EAF2F8",
  surfaceMuted: "#DDEAF3",
  border: "#B9CDDC",
  text: "#102231",
  textMuted: "#526B7E",
  primary: "#007EBD",
  primarySoft: "#D8F0FC",
  positive: "#008C68",
  positiveSoft: "#D8F5EC",
  negative: "#C23D57",
  negativeSoft: "#FCE5EA",
  warning: "#9A6800",
  warningSoft: "#FFF2CC",
  white: "#FFFFFF",
};

/** Mutable compatibility tokens for existing screen styles; ThemeProvider synchronizes them before remounting its tree. */
export const palette: { -readonly [Key in keyof ThemePalette]: string } = { ...darkPalette };

export type ResolvedTheme = "dark" | "light";
export type ThemeTokens = ThemePalette & { readonly mode: ResolvedTheme };
export type QuoteColors = {
  up: string;
  upSoft: string;
  down: string;
  downSoft: string;
};

export const spacing = {
  xxs: 4,
  xs: 8,
  sm: 12,
  md: 16,
  lg: 24,
  xl: 32,
} as const;

export const radius = {
  sm: 10,
  md: 16,
  lg: 22,
  pill: 999,
} as const;

export function resolveTheme(
  preference: MobileThemePreference,
  systemColorScheme: ColorSchemeName | null = null,
): ResolvedTheme {
  if (preference !== "system") return preference;
  return systemColorScheme === "light" ? "light" : "dark";
}

export function resolveThemeTokens(
  preference: MobileThemePreference,
  systemColorScheme: ColorSchemeName | null = null,
): ThemeTokens {
  const mode = resolveTheme(preference, systemColorScheme);
  return { ...(mode === "dark" ? darkPalette : lightPalette), mode };
}

export function quoteColorsFor(
  stockColorMode: StockColorMode,
  tokens: ThemePalette = darkPalette,
): QuoteColors {
  const greenUp = stockColorMode === "GREEN_UP_RED_DOWN";
  return {
    up: greenUp ? tokens.positive : tokens.negative,
    upSoft: greenUp ? tokens.positiveSoft : tokens.negativeSoft,
    down: greenUp ? tokens.negative : tokens.positive,
    downSoft: greenUp ? tokens.negativeSoft : tokens.positiveSoft,
  };
}
