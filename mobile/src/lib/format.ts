export function formatGloss(label: string | null | undefined): string {
  if (!label) return "—";
  return label.replace(/_/g, " ");
}

export function formatDisplay(label: string | null | undefined): string {
  if (!label) return "—";
  return label
    .split("_")
    .map((w) => w.charAt(0) + w.slice(1).toLowerCase())
    .join(" ");
}
