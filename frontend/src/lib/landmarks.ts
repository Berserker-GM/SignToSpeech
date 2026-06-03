/** Match utils/landmarks.py normalization for model input. */
export function extractLandmarks(
  landmarks: { x: number; y: number; z: number }[]
): number[] {
  if (landmarks.length < 21) return [];

  const wrist = landmarks[0];
  const coords: number[][] = [];

  for (const point of landmarks) {
    coords.push([
      point.x - wrist.x,
      point.y - wrist.y,
      point.z - wrist.z,
    ]);
  }

  const tip = coords[12];
  const scale = Math.hypot(tip[0], tip[1], tip[2]);
  const normalized =
    scale > 0
      ? coords.map(([x, y, z]) => [x / scale, y / scale, z / scale])
      : coords;

  return normalized.flat();
}

export function formatGloss(label: string | null): string {
  if (!label) return "—";
  return label.replace(/_/g, " ");
}

export function formatDisplay(label: string | null): string {
  if (!label) return "—";
  return label
    .split("_")
    .map((w) => w.charAt(0) + w.slice(1).toLowerCase())
    .join(" ");
}
