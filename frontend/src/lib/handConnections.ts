/** MediaPipe hand landmark connections (same topology as Python HAND_CONNECTIONS). */
export const HAND_CONNECTIONS: [number, number][] = [
  [0, 1], [1, 2], [2, 3], [3, 4],
  [0, 5], [5, 6], [6, 7], [7, 8],
  [0, 9], [9, 10], [10, 11], [11, 12],
  [0, 13], [13, 14], [14, 15], [15, 16],
  [0, 17], [17, 18], [18, 19], [19, 20],
  [5, 9], [9, 13], [13, 17],
];

export type LandmarkPoint = { x: number; y: number; z: number };

/** Mirror x to match Python cv2.flip(frame, 1) before MediaPipe. */
export function mirrorLandmarks(landmarks: LandmarkPoint[]): LandmarkPoint[] {
  return landmarks.map((p) => ({ x: 1 - p.x, y: p.y, z: p.z }));
}

export function drawHandOverlay(
  ctx: CanvasRenderingContext2D,
  landmarks: LandmarkPoint[],
  width: number,
  height: number
): void {
  ctx.clearRect(0, 0, width, height);

  const px = (p: LandmarkPoint) => ({
    x: p.x * width,
    y: p.y * height,
  });

  const points = landmarks.map(px);

  // Connections (lime green lines like MediaPipe)
  ctx.lineWidth = 3;
  ctx.strokeStyle = "rgba(0, 255, 136, 0.95)";
  ctx.lineCap = "round";
  for (const [a, b] of HAND_CONNECTIONS) {
    const p1 = points[a];
    const p2 = points[b];
    ctx.beginPath();
    ctx.moveTo(p1.x, p1.y);
    ctx.lineTo(p2.x, p2.y);
    ctx.stroke();
  }

  // Joint dots
  for (let i = 0; i < points.length; i++) {
    const p = points[i];
    const r = i === 0 ? 7 : i % 4 === 0 ? 5 : 4;
    ctx.beginPath();
    ctx.arc(p.x, p.y, r, 0, Math.PI * 2);
    ctx.fillStyle = i === 0 ? "#ff6b6b" : "rgba(139, 92, 246, 0.95)";
    ctx.fill();
    ctx.strokeStyle = "rgba(255,255,255,0.9)";
    ctx.lineWidth = 1.5;
    ctx.stroke();
  }
}
