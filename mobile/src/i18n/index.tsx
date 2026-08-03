import { createContext, type PropsWithChildren, useCallback, useContext, useMemo } from "react";
import { usePreferences, type MobileLanguage } from "../preferences";
import { en } from "./locales/en";
import { ja } from "./locales/ja";
import { zh_CN } from "./locales/zh_CN";
import { zh_TW } from "./locales/zh_TW";

export type TranslationKey = keyof typeof en;
export type TranslationValues = Readonly<Record<string, string | number>>;
type TranslationDictionary = Record<TranslationKey, string>;

const resources: Readonly<Record<MobileLanguage, TranslationDictionary>> = {
  en,
  zh_CN,
  zh_TW,
  ja,
};

function interpolate(template: string, values?: TranslationValues): string {
  if (!values) return template;
  return template.replace(/\{(\w+)\}/g, (match, key: string) => {
    const value = values[key];
    return value == null ? match : String(value);
  });
}

export function translate(
  language: MobileLanguage,
  key: TranslationKey,
  values?: TranslationValues,
): string {
  return interpolate(resources[language][key] ?? resources.en[key], values);
}

type I18nContextValue = {
  language: MobileLanguage;
  t: (key: TranslationKey, values?: TranslationValues) => string;
};

const I18nContext = createContext<I18nContextValue | null>(null);

export function I18nProvider({ children }: PropsWithChildren) {
  const { preferences } = usePreferences();
  const language = preferences.language;
  const t = useCallback(
    (key: TranslationKey, values?: TranslationValues) => translate(language, key, values),
    [language],
  );
  const value = useMemo<I18nContextValue>(() => ({ language, t }), [language, t]);
  return <I18nContext.Provider value={value}>{children}</I18nContext.Provider>;
}

export function useI18n(): I18nContextValue {
  const value = useContext(I18nContext);
  if (!value) throw new Error("useI18n must be used inside I18nProvider");
  return value;
}
