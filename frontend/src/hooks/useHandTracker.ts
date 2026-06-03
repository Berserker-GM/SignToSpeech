import { useCallback, useEffect, useRef, useState } from "react";
import {
  FilesetResolver,
  HandLandmarker,
  type HandLandmarkerResult,
} from "@mediapipe/tasks-vision";
import { extractLandmarks } from "../lib/landmarks";
import {
  drawHandOverlay,
  mirrorLandmarks,
  type LandmarkPoint,
} from "../lib/handConnections";

export function useHandTracker(
  videoRef: React.RefObject<HTMLVideoElement | null>,
  canvasRef: React.RefObject<HTMLCanvasElement | null>
) {
  const landmarkerRef = useRef<HandLandmarker | null>(null);
  const processCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const rafRef = useRef<number>(0);
  const [ready, setReady] = useState(false);
  const [landmarks, setLandmarks] = useState<number[] | null>(null);
  const [handDetected, setHandDetected] = useState(false);

  useEffect(() => {
    let cancelled = false;

    (async () => {
      const vision = await FilesetResolver.forVisionTasks(
        "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.18/wasm"
      );
      const landmarker = await HandLandmarker.createFromOptions(vision, {
        baseOptions: {
          modelAssetPath:
            "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
          delegate: "GPU",
        },
        runningMode: "VIDEO",
        numHands: 1,
        minHandDetectionConfidence: 0.7,
        minHandPresenceConfidence: 0.5,
        minTrackingConfidence: 0.5,
      });
      if (!cancelled) {
        landmarkerRef.current = landmarker;
        setReady(true);
      }
    })();

    return () => {
      cancelled = true;
      landmarkerRef.current?.close();
    };
  }, []);

  const syncCanvasSize = useCallback(() => {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (!video || !canvas) return;

    const w = video.clientWidth;
    const h = video.clientHeight;
    if (canvas.width !== w || canvas.height !== h) {
      canvas.width = w;
      canvas.height = h;
    }
  }, [videoRef, canvasRef]);

  /** Same as cv2.flip(frame, 1) before MediaPipe in Python. */
  const detectOnFlippedFrame = useCallback(
    (video: HTMLVideoElement, landmarker: HandLandmarker): HandLandmarkerResult => {
      const vw = video.videoWidth;
      const vh = video.videoHeight;
      if (!vw || !vh) {
        return landmarker.detectForVideo(video, performance.now());
      }

      if (!processCanvasRef.current) {
        processCanvasRef.current = document.createElement("canvas");
      }
      const proc = processCanvasRef.current;
      proc.width = vw;
      proc.height = vh;
      const pctx = proc.getContext("2d");
      if (!pctx) {
        return landmarker.detectForVideo(video, performance.now());
      }

      pctx.clearRect(0, 0, vw, vh);
      pctx.save();
      pctx.translate(vw, 0);
      pctx.scale(-1, 1);
      pctx.drawImage(video, 0, 0, vw, vh);
      pctx.restore();

      return landmarker.detectForVideo(proc, performance.now());
    },
    []
  );

  const loop = useCallback(() => {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    const landmarker = landmarkerRef.current;

    if (!video || !landmarker || video.readyState < 2) {
      rafRef.current = requestAnimationFrame(loop);
      return;
    }

    syncCanvasSize();

    const result = detectOnFlippedFrame(video, landmarker);
    const ctx = canvas?.getContext("2d");

    if (result.landmarks?.[0]) {
      const fromFlipped = result.landmarks[0];

      // Landmarks match training (flipped frame). Use directly for the model.
      setLandmarks(extractLandmarks(fromFlipped));
      setHandDetected(true);

      // Map back onto the natural (unmirrored) camera view for the overlay.
      if (ctx && canvas) {
        drawHandOverlay(
          ctx,
          mirrorLandmarks(fromFlipped),
          canvas.width,
          canvas.height
        );
      }
    } else {
      setLandmarks(null);
      setHandDetected(false);
      if (ctx && canvas) {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
      }
    }

    rafRef.current = requestAnimationFrame(loop);
  }, [videoRef, canvasRef, syncCanvasSize, detectOnFlippedFrame]);

  useEffect(() => {
    if (!ready) return;
    rafRef.current = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(rafRef.current);
  }, [ready, loop]);

  useEffect(() => {
    const video = videoRef.current;
    if (!video) return;
    const ro = new ResizeObserver(() => syncCanvasSize());
    ro.observe(video);
    return () => ro.disconnect();
  }, [videoRef, syncCanvasSize, ready]);

  return { ready, landmarks, handDetected };
}
