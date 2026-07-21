export function initialCapitalLabel(value: number | undefined): string {
  return value === undefined ? "—" : `${value.toLocaleString()} USDT`;
}
