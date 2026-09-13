/**
 * In-WebView MediaPipe hand tracker.
 * Matches frontend useHandTracker + utils/landmarks.py so landmarks
 * work with the same STS models (sign_model.pkl / dynamic_model.keras).
 */
export const HAND_TRACKER_HTML = `<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1, user-scalable=no" />
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    html, body { width: 100%; height: 100%; background: #0f1115; overflow: hidden; }
    #stage { position: relative; width: 100%; height: 100%; background: #0f1115; }
    video, canvas {
      position: absolute; inset: 0;
      width: 100%; height: 100%;
      object-fit: cover;
    }
    canvas { pointer-events: none; z-index: 2; }
    #status {
      position: absolute; left: 12px; bottom: 12px; z-index: 3;
      color: rgba(255,255,255,0.85); font: 600 12px/1.3 -apple-system, system-ui, sans-serif;
      background: rgba(0,0,0,0.45); padding: 6px 10px; border-radius: 8px;
    }
  </style>
</head>
<body>
  <div id="stage">
    <video id="video" playsinline muted autoplay></video>
    <canvas id="overlay"></canvas>
    <div id="status">Starting camera…</div>
  </div>
  <script type="module">
    import {
      FilesetResolver,
      HandLandmarker,
    } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.18";

    const CONNECTIONS = [
      [0,1],[1,2],[2,3],[3,4],
      [0,5],[5,6],[6,7],[7,8],
      [0,9],[9,10],[10,11],[11,12],
      [0,13],[13,14],[14,15],[15,16],
      [0,17],[17,18],[18,19],[19,20],
      [5,9],[9,13],[13,17],
    ];

    const video = document.getElementById("video");
    const overlay = document.getElementById("overlay");
    const statusEl = document.getElementById("status");
    const proc = document.createElement("canvas");
    let landmarker = null;
    let stream = null;
    let running = true;
    let raf = 0;

    function post(payload) {
      if (window.ReactNativeWebView) {
        window.ReactNativeWebView.postMessage(JSON.stringify(payload));
      }
    }

    function setStatus(text) {
      statusEl.textContent = text;
    }

    function extractLandmarks(landmarks) {
      if (!landmarks || landmarks.length < 21) return [];
      const wrist = landmarks[0];
      const coords = landmarks.map((p) => [
        p.x - wrist.x,
        p.y - wrist.y,
        p.z - wrist.z,
      ]);
      const tip = coords[12];
      const scale = Math.hypot(tip[0], tip[1], tip[2]);
      const normalized =
        scale > 0
          ? coords.map(([x, y, z]) => [x / scale, y / scale, z / scale])
          : coords;
      return normalized.flat();
    }

    function mirrorLandmarks(landmarks) {
      return landmarks.map((p) => ({ x: 1 - p.x, y: p.y, z: p.z }));
    }

    function drawHand(ctx, landmarks, w, h) {
      ctx.clearRect(0, 0, w, h);
      const pts = landmarks.map((p) => ({ x: p.x * w, y: p.y * h }));
      ctx.lineWidth = 3;
      ctx.strokeStyle = "rgba(0, 255, 136, 0.95)";
      ctx.lineCap = "round";
      for (const [a, b] of CONNECTIONS) {
        ctx.beginPath();
        ctx.moveTo(pts[a].x, pts[a].y);
        ctx.lineTo(pts[b].x, pts[b].y);
        ctx.stroke();
      }
      for (let i = 0; i < pts.length; i++) {
        const p = pts[i];
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

    function syncOverlaySize() {
      const w = overlay.clientWidth || window.innerWidth;
      const h = overlay.clientHeight || window.innerHeight;
      if (overlay.width !== w || overlay.height !== h) {
        overlay.width = w;
        overlay.height = h;
      }
    }

    function detectOnFlippedFrame() {
      const vw = video.videoWidth;
      const vh = video.videoHeight;
      if (!vw || !vh) {
        return landmarker.detectForVideo(video, performance.now());
      }
      proc.width = vw;
      proc.height = vh;
      const pctx = proc.getContext("2d");
      pctx.clearRect(0, 0, vw, vh);
      pctx.save();
      pctx.translate(vw, 0);
      pctx.scale(-1, 1);
      pctx.drawImage(video, 0, 0, vw, vh);
      pctx.restore();
      return landmarker.detectForVideo(proc, performance.now());
    }

    function loop() {
      if (!running || !landmarker || video.readyState < 2) {
        raf = requestAnimationFrame(loop);
        return;
      }
      syncOverlaySize();
      const result = detectOnFlippedFrame();
      const ctx = overlay.getContext("2d");
      if (result.landmarks && result.landmarks[0]) {
        const fromFlipped = result.landmarks[0];
        const features = extractLandmarks(fromFlipped);
        post({ type: "landmarks", landmarks: features, handDetected: true });
        drawHand(ctx, mirrorLandmarks(fromFlipped), overlay.width, overlay.height);
        setStatus("Hand detected");
      } else {
        post({ type: "landmarks", landmarks: null, handDetected: false });
        ctx.clearRect(0, 0, overlay.width, overlay.height);
        setStatus("Show your hand");
      }
      raf = requestAnimationFrame(loop);
    }

    async function startCamera() {
      try {
        if (stream) {
          stream.getTracks().forEach((t) => t.stop());
        }
        stream = await navigator.mediaDevices.getUserMedia({
          video: {
            facingMode: "user",
            width: { ideal: 1280 },
            height: { ideal: 720 },
          },
          audio: false,
        });
        video.srcObject = stream;
        await video.play();
        post({ type: "camera", status: "on" });
        setStatus("Camera on");
      } catch (e) {
        post({ type: "camera", status: "denied", message: String(e) });
        setStatus("Camera permission denied");
      }
    }

    function stopCamera() {
      if (stream) {
        stream.getTracks().forEach((t) => t.stop());
        stream = null;
      }
      video.srcObject = null;
      post({ type: "camera", status: "off" });
      setStatus("Camera off");
    }

    async function init() {
      try {
        setStatus("Loading hand tracker…");
        const vision = await FilesetResolver.forVisionTasks(
          "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.18/wasm"
        );
        landmarker = await HandLandmarker.createFromOptions(vision, {
          baseOptions: {
            modelAssetPath:
              "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
            delegate: "GPU",
          },
          runningMode: "VIDEO",
          numHands: 1,
          minHandDetectionConfidence: 0.60,
          minHandPresenceConfidence: 0.60,
          minTrackingConfidence: 0.60,
        });
        post({ type: "ready" });
        await startCamera();
        running = true;
        raf = requestAnimationFrame(loop);
      } catch (e) {
        post({ type: "error", message: String(e) });
        setStatus("Tracker failed to load");
      }
    }

    document.addEventListener("message", onBridge);
    window.addEventListener("message", onBridge);

    function onBridge(event) {
      let data = event.data;
      try {
        if (typeof data === "string") data = JSON.parse(data);
      } catch (_) {
        return;
      }
      if (!data || !data.type) return;
      if (data.type === "pause") running = false;
      if (data.type === "resume") {
        running = true;
        cancelAnimationFrame(raf);
        raf = requestAnimationFrame(loop);
      }
      if (data.type === "startCamera") startCamera();
      if (data.type === "stopCamera") stopCamera();
    }

    init();
  </script>
</body>
</html>`;
