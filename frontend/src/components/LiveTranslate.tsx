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
  Bookmark,
  Check,
} from "lucide-react";
import { useHandTracker } from "../hooks/useHandTracker";
import { useLiveSession } from "../hooks/useLiveSession";
import { formatDisplay, formatGloss } from "../lib/landmarks";
import { correctSentence } from "../lib/api";
import { saveHistoryEntry } from "../lib/history";

type Props = {
  onWordSpoken: (word: string) => void;
  onSentenceChange: (words: string[]) => void;
  registerClear?: (fn: () => void) => void;
  onClear?: () => void;
  recognitionMode?: "AUTO" | "STATIC" | "DYNAMIC";
  onSavedToHistory?: () => void;
};

export function LiveTranslate({
  onWordSpoken,
  onSentenceChange,
  registerClear,
  onClear,
  recognitionMode = "AUTO",
  onSavedToHistory,
}: Props) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [cameraOn, setCameraOn] = useState(false);
  const [paused, setPaused] = useState(false);
  const [showRaw, setShowRaw] = useState(false);
  const [translated, setTranslated] = useState("—");
  const [rawSentence, setRawSentence] = useState("");
  const [grammarActive, setGrammarActive] = useState(true);
  const [grammarError, setGrammarError] = useState<string | null>(null);
  const [historySaved, setHistorySaved] = useState(false);
  const lastConfirmed = useRef<string | null>(null);

  const { ready, landmarks, handDetected } = useHandTracker(videoRef, canvasRef);
  const { connected, frame, sentence, mode, serverVocab, sendLandmarks, clearSentence } =
    useLiveSession(cameraOn && !paused, recognitionMode);

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
        setHistorySaved(false);
        return;
      }
      setHistorySaved(false);
      const raw = sentence.join(" ");
      setRawSentence(raw);
      try {
        const r = await correctSentence(sentence);
        setTranslated(r.corrected);
        setGrammarActive(r.grammar_active);
        setGrammarError(r.grammar_active ? null : r.error || "Grammar offline");
      } catch {
        setTranslated(sentence.map((w) => formatDisplay(w)).join(" "));
        setGrammarActive(false);
        setGrammarError("Grammar request failed");
      }
    };
    const t = setTimeout(run, 400);
    return () => clearTimeout(t);
  }, [sentence]);

  const saveToHistory = () => {
    if (!sentence.length) return;
    const corrected =
      translated && translated !== "—"
        ? translated
        : sentence.map((w) => formatDisplay(w)).join(" ");
    const raw = rawSentence || sentence.join(" ");
    saveHistoryEntry({ raw, corrected, words: [...sentence] });
    setHistorySaved(true);
    onSavedToHistory?.();
  };

  // Prefer locked final label; fall back to raw static so UI matches main.py feedback
  const detected = frame.final_pred || (recognitionMode === "STATIC" ? frame.static_pred : null);
  const conf = frame.final_pred
    ? (frame.final_conf ?? 0)
    : (frame.static_conf ?? 0);
  const hasSign = Boolean(handDetected && detected && conf >= 0.35);
  const displaySign = hasSign ? formatDisplay(detected ?? null) : "—";
  const gloss = hasSign ? formatGloss(detected ?? null) : "—";
  const sentenceDisplay = showRaw
    ? rawSentence || "—"
    : translated;
  const hasSentence = sentence.length > 0 && sentenceDisplay !== "—";
  const isDetecting =
    cameraOn &&
    !paused &&
    handDetected &&
    ((frame.hold_progress != null && frame.hold_progress > 0) || !!detected);

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

      <div
        className={`video-stage ${!cameraOn ? "empty" : ""} ${isDetecting ? "detecting" : ""}`}
      >
        <div className="video-stack">
          <video ref={videoRef} playsInline muted />
          <canvas ref={canvasRef} className="hand-overlay" aria-hidden />
        </div>
        {!cameraOn && (
          <div className="video-placeholder">
            <CameraOff size={48} strokeWidth={1.5} />
            <span>Camera is off — allow access to start</span>
            <button
              type="button"
              className="btn-speak"
              style={{ maxWidth: 200 }}
              onClick={startCamera}
            >
              Turn on camera
            </button>
          </div>
        )}
        {cameraOn && (
          <div className="video-overlay">
            {!paused && (
              <span className="live-indicator">
                <span className="pulse" aria-hidden />
                LIVE
              </span>
            )}
            <button type="button" className="icon-btn fs-btn" aria-label="Fullscreen">
              <Maximize2 size={16} />
            </button>
            {frame.hold_progress != null && frame.hold_progress > 0 && (
              <svg className="hold-ring" viewBox="0 0 48 48">
                <circle
                  cx="24"
                  cy="24"
                  r="20"
                  fill="none"
                  stroke="rgba(255,255,255,0.25)"
                  strokeWidth="4"
                />
                <circle
                  cx="24"
                  cy="24"
                  r="20"
                  fill="none"
                  stroke="#34d399"
                  strokeWidth="4"
                  strokeLinecap="round"
                  strokeDasharray={`${frame.hold_progress * 126} 126`}
                  transform="rotate(-90 24 24)"
                />
              </svg>
            )}
            <div className="video-toolbar">
              <button type="button" className="video-ctrl" title="Capture">
                <Image size={20} strokeWidth={2} />
              </button>
              <button
                type="button"
                className="video-ctrl primary"
                onClick={() => setPaused((p) => !p)}
                title={paused ? "Resume" : "Pause"}
              >
                {paused ? (
                  <Play size={22} fill="white" />
                ) : (
                  <Pause size={22} fill="white" />
                )}
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
              <div
                key={hasSign ? displaySign : "empty"}
                className={`det-sign-word ${!hasSign ? "muted" : "text-arrive"}`}
              >
                {displaySign}
              </div>
              {!hasSign && (
                <p className="empty-hint">Hold a sign in frame to detect</p>
              )}
              {conf > 0 && hasSign && (
                <span className="conf-badge">
                  {Math.round(conf * 100)}% Confidence
                </span>
              )}
            </div>
          </div>
          <p className="gloss-line">Gloss: {gloss}</p>
        </div>

        <div className="card det-card">
          <div className="det-card-label">Translated Sentence</div>
          <div
            key={hasSentence ? sentenceDisplay : "empty-sent"}
            className={`det-sentence-text ${!hasSentence ? "muted" : "text-arrive"}`}
          >
            {sentenceDisplay}
          </div>
          {!hasSentence && (
            <p className="empty-hint">Confirmed signs will build a sentence here</p>
          )}
          {grammarActive && hasSentence && !showRaw && (
            <span className="grammar-tag">
              <Sparkles size={12} /> Grammar Corrected
            </span>
          )}
          <div className="det-card-actions">
            <button
              type="button"
              className="show-raw-link"
              onClick={() => setShowRaw((s) => !s)}
            >
              {showRaw ? "Show corrected sentence" : "Show Original (Raw)"}
            </button>
            <button
              type="button"
              className={`show-raw-link save-history-link ${historySaved ? "saved" : ""}`}
              onClick={saveToHistory}
              disabled={!hasSentence || historySaved}
              title={
                hasSentence
                  ? "Save this conversation to History"
                  : "Confirm signs first"
              }
            >
              {historySaved ? (
                <>
                  <Check size={14} strokeWidth={2.5} /> Saved
                </>
              ) : (
                <>
                  <Bookmark size={14} strokeWidth={2.25} /> Save in history
                </>
              )}
            </button>
          </div>
        </div>
      </div>

      <div className="card grammar-bar">
        <div>
          <div className="grammar-bar-title">AI Grammar Correction</div>
          <div className="grammar-bar-desc">
            {grammarError
              ? grammarError
              : "Grammar and punctuation improved for clarity"}
          </div>
        </div>
        <span className={`badge-pill ${grammarActive ? "" : "offline"}`}>
          {grammarActive ? "Active" : "Offline"}
        </span>
      </div>

      <div className="footer-bar">
        <select className="lang-select" defaultValue="en-US" aria-label="Language">
          <option value="en-US">English (US)</option>
        </select>
        <div className="realtime-pill">
          <Activity size={18} color="var(--accent)" strokeWidth={2.5} />
          <span>
            <strong>{mode || recognitionMode}</strong>
            {connected ? "" : " · API offline"}
            {connected && serverVocab.length > 0
              ? ` · ${serverVocab.length} signs loaded`
              : ""}
            {" — "}
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
