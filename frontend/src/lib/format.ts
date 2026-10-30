// Display formatting only.

export function ms(value: number | null | undefined): string {
  if (value == null) return "–";
  return value >= 1000 ? `${(value / 1000).toFixed(2)} s` : `${Math.round(value)} ms`;
}

export function percent(value: number | null | undefined, digits = 0): string {
  return value == null ? "–" : `${(value * 100).toFixed(digits)}%`;
}

export function usd(value: number | null | undefined): string {
  if (value == null) return "–";
  return value < 0.01 ? `$${value.toFixed(4)}` : `$${value.toFixed(2)}`;
}

export function count(value: number | null | undefined): string {
  return value == null ? "–" : value.toLocaleString("en-IN");
}

export function when(iso: string | null | undefined): string {
  if (!iso) return "–";
  const date = new Date(iso);
  return Number.isNaN(date.getTime())
    ? iso
    : date.toLocaleString(undefined, { dateStyle: "medium", timeStyle: "short" });
}

export function title(snake: string): string {
  const words = snake.replaceAll("_", " ");
  return words.charAt(0).toUpperCase() + words.slice(1);
}
