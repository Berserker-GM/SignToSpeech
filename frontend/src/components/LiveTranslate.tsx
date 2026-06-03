import { useCallback, useEffect, useRef, useState } from "react";
import {
  MoreVertical,
  Camera,
  CameraOff,
  Image,
  Lightbulb,
  Pause,
  Play,
  Hand,
  Sparkles,
  Activity,
  Maximize2,
} from "lucide-react";
import { useHandTracker } from "../hooks/useHandTracker";
import { useLiveSession } from "../hooks/useLiveSession";
import { formatDisplay, formatGloss } from "../lib/landmarks";
import { correctSentence } from "../lib/api";

type Props = {
  onWordSpoken: (word: string) => void;
  onSentenceChange: (words: string[]) => void;
  registerClear?: (fn: () => void) => void;
  onClear?: () => void;
};

export function LiveTranslate({
  onWordSpoken,
  onSentenceChange,
  registerClear,
  onClear,
}: Props) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [cameraOn, setCameraOn] = useState(false);
  const [paused, setPaused] = useState(false);
  const [showRaw, setShowRaw] = useState(false);
  const [translated, setTranslated] = useState("—");
  const [rawSentence, setRawSentence] = useState("");
  const [grammarActive, setGrammarActive] = useState(true);
  const lastConfirmed = useRef<string | null>(null);

  const { ready, landmarks, handDetected } = useHandTracker(videoRef, canvasRef);
  const { connected, frame, sentence, sendLandmarks, clearSentence } = useLiveSession(
    cameraOn && !paused
  );

  const startCamera = useCallback(async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: 1280, height: 720, facingMode: "user" },
        audio: false,
      });
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
      }
      setCameraOn(true);
    } catch {
      setCameraOn(false);
    }
  }, []);

  const stopCamera = () => {
    const stream = videoRef.current?.srcObject as MediaStream | null;
    stream?.getTracks().forEach((t) => t.stop());
    if (videoRef.current) videoRef.current.srcObject = null;
    setCameraOn(false);
  };

  useEffect(() => {
    startCamera();
    return () => stopCamera();
  }, [startCamera]);

  useEffect(() => {
    if (!connected || paused) return;
    if (landmarks && handDetected) sendLandmarks(landmarks);
    else sendLandmarks(null);
  }, [landmarks, handDetected, connected, paused, sendLandmarks]);

  useEffect(() => {
    onSentenceChange(sentence);
  }, [sentence, onSentenceChange]);

  useEffect(() => {
    registerClear?.(() => {
      clearSentence();
      onSentenceChange([]);
      setTranslated("—");
      setRawSentence("");
      setShowRaw(false);
      lastConfirmed.current = null;
    });
  }, [registerClear, clearSentence, onSentenceChange]);

  useEffect(() => {
    if (frame.confirmed && frame.confirmed !== lastConfirmed.current) {
      lastConfirmed.current = frame.confirmed;
      onWordSpoken(frame.confirmed);
    }
  }, [frame.confirmed, onWordSpoken]);

  useEffect(() => {
    const run = async () => {
      if (!sentence.length) {
        setTranslated("—");
        setRawSentence("");
        return;
      }
      const raw = sentence.join(" ");
      setRawSentence(raw);
      try {
        const r = await correctSentence(sentence);
        setTranslated(r.corrected);
        setGrammarActive(r.grammar_active);
      } catch {
        setTranslated(sentence.map((w) => formatDisplay(w)).join(" "));
      }
    };
    const t = setTimeout(run, 400);
    return () => clearTimeout(t);
  }, [sentence]);

  const detected = frame.final_pred;
  const conf = frame.final_conf ?? 0;
  const hasSign = handDetected && detected;
  const displaySign = hasSign ? formatDisplay(detected) : "—";
  const gloss = hasSign ? formatGloss(detected) : "—";
  const sentenceDisplay = showRaw
    ? rawSentence || "—"
    : translated;
  const hasSentence = sentence.length > 0 && sentenceDisplay !== "—";

  return (
    <div className="center-column">
      <header className="page-header">
        <div className="page-header-text">
          <h1>
            Live Sign Detection
            {cameraOn && !paused && <span className="live-dot" aria-hidden />}
          </h1>
          <p>Real-time sign language recognition and translation</p>
        </div>
        <div className="page-header-actions">
          <span className={`status-badge ${cameraOn ? "active" : ""}`}>
            {cameraOn ? <Camera size={15} /> : <CameraOff size={15} />}
            {cameraOn ? "Camera Active" : "Camera Off"}
          </span>
          <button type="button" className="icon-btn" aria-label="More options">
            <MoreVertical size={18} />
          </button>
        </div>
      </header>

      <div className={`video-stage ${!cameraOn ? "empty" : ""}`}>
        <div className="video-stack">
          <video ref={videoRef} playsInline muted />
          <canvas ref={canvasRef} className="hand-overlay" aria-hidden />
        </div>
        {!cameraOn && (
          <div className="video-placeholder">
            <CameraOff size={48} strokeWidth={1.5} />
            <span>Camera is off — allow access to start</span>
            <button type="button" className="btn-speak" style={{ maxWidth: 200 }} onClick={startCamera}>
              Turn on camera
            </button>
          </div>
        )}
        {cameraOn && (
          <div className="video-overlay">
            {!paused && <span className="live-badge">LIVE</span>}
            <button
              type="button"
              className="icon-btn"
              style={{
                position: "absolute",
                top: 16,
                right: 16,
                pointerEvents: "auto",
                background: "rgba(255,255,255,0.9)",
              }}
              aria-label="Fullscreen"
            >
              <Maximize2 size={16} />
            </button>
            {frame.hold_progress != null && frame.hold_progress > 0 && (
              <svg className="hold-ring" viewBox="0 0 48 48">
                <circle cx="24" cy="24" r="20" fill="none" stroke="rgba(255,255,255,0.25)" strokeWidth="4" />
                <circle
                  cx="24"
                  cy="24"
                  r="20"
                  fill="none"
                  stroke="#22c55e"
                  strokeWidth="4"
                  strokeLinecap="round"
                  strokeDasharray={`${frame.hold_progress * 126} 126`}
                  transform="rotate(-90 24 24)"
                />
              </svg>
            )}
            <div className="video-controls">
              <button type="button" className="video-ctrl" title="Capture">
                <Image size={20} strokeWidth={2} />
              </button>
              <button
                type="button"
                className="video-ctrl primary"
                onClick={() => setPaused((p) => !p)}
                title={paused ? "Resume" : "Pause"}
              >
                {paused ? <Play size={24} fill="white" /> : <Pause size={24} fill="white" />}
              </button>
              <button type="button" className="video-ctrl" title="Tips">
                <Lightbulb size={20} strokeWidth={2} />
              </button>
            </div>
          </div>
        )}
        {cameraOn && !ready && <div className="video-loading">Loading hand tracker…</div>}
      </div>

      <div className="detection-row">
        <div className="card det-card">
          <div className="det-card-label">Detected Sign</div>
          <div className="det-sign-row">
            <div className="det-icon-wrap">
              <Hand size={22} strokeWidth={2} />
            </div>
            <div className="det-sign-main">
              <div className={`det-sign-word ${!hasSign ? "muted" : ""}`}>{displaySign}</div>
              {conf > 0 && hasSign && (
                <span className="conf-badge">{Math.round(conf * 100)}% Confidence</span>
              )}
            </div>
          </div>
          <p className="gloss-line">Gloss: {gloss}</p>
        </div>

        <div className="card det-card">
          <div className="det-card-label">Translated Sentence</div>
          <div className={`det-sentence-text ${!hasSentence ? "muted" : ""}`}>
            {sentenceDisplay}
          </div>
          {grammarActive && hasSentence && !showRaw && (
            <span className="grammar-tag">
              <Sparkles size={12} /> Grammar Corrected
            </span>
          )}
          <button type="button" className="show-raw-link" onClick={() => setShowRaw((s) => !s)}>
            {showRaw ? "Show corrected sentence" : "Show Original (Raw)"}
          </button>
        </div>
      </div>

      <div className="card grammar-bar">
        <div>
          <div className="grammar-bar-title">AI Grammar Correction</div>
          <div className="grammar-bar-desc">Grammar and punctuation improved for clarity</div>
        </div>
        <span className="badge-pill">{grammarActive ? "Active" : "Offline"}</span>
      </div>

      <div className="footer-bar">
        <select className="lang-select" defaultValue="en-US" aria-label="Language">
          <option value="en-US">English (US)</option>
        </select>
        <div className="realtime-pill">
          <Activity size={18} color="var(--purple)" strokeWidth={2.5} />
          <span>
            <strong>Real-time Mode</strong> —{" "}
            {connected && cameraOn && !paused
              ? "System is actively listening"
              : "Paused"}
          </span>
        </div>
        <div className="info-box">
          Perfect for hospitals, buses, schools, public services and more.
        </div>
        {sentence.length > 0 && onClear && (
          <button type="button" className="btn-text-sm" onClick={onClear}>
            Clear all
          </button>
        )}
      </div>
    </div>
  );
}
