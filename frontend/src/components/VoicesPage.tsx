import { useEffect, useState } from "react";
import { Mic2, Play, Sparkles } from "lucide-react";
import { fetchVoices } from "../lib/api";
import { speakBrowser, stopBrowserSpeech } from "../lib/browserTts";
import {
  DEMO_VOICES,
  loadSelectedVoiceId,
  mergeVoiceLists,
  saveSelectedVoiceId,
  type ShowcaseVoice,
} from "../lib/voices";

export function VoicesPage() {
  const [voices, setVoices] = useState<ShowcaseVoice[]>(DEMO_VOICES);
  const [voiceId, setVoiceId] = useState(loadSelectedVoiceId);
  const [previewing, setPreviewing] = useState<string | null>(null);

  useEffect(() => {
    fetchVoices()
      .then(({ voices: api, defaultId }) => {
        setVoices(mergeVoiceLists(api));
        const saved = loadSelectedVoiceId(defaultId);
        setVoiceId(saved);
      })
      .catch(() => {
        setVoices(DEMO_VOICES);
      });
    if ("speechSynthesis" in window) {
      speechSynthesis.getVoices();
    }
  }, []);

  const select = (id: string) => {
    setVoiceId(id);
    saveSelectedVoiceId(id);
  };

  const preview = async (v: ShowcaseVoice) => {
    stopBrowserSpeech();
    setPreviewing(v.id);
    try {
      await speakBrowser(
        `Hello, I am ${v.name}. ${v.description}`,
        0.85
      );
    } catch {
      /* ignore preview failures */
    } finally {
      setPreviewing(null);
    }
  };

  return (
    <div className="voices-page">
      <header className="page-header">
        <div className="page-header-text">
          <h1>Voices</h1>
          <p>Pick a speaking voice for Live Translate. Sample voices are demos.</p>
        </div>
      </header>

      <div className="voices-grid">
        {voices.map((v) => {
          const selected = voiceId === v.id;
          return (
            <article
              key={v.id}
              className={`card voice-card ${selected ? "selected" : ""}`}
              onClick={() => select(v.id)}
              onKeyDown={(e) => {
                if (e.key === "Enter" || e.key === " ") {
                  e.preventDefault();
                  select(v.id);
                }
              }}
              role="radio"
              aria-checked={selected}
              tabIndex={0}
            >
              <div className="voice-card-top">
                <div className="voice-avatar" aria-hidden>
                  {v.name[0]}
                </div>
                <div className="voice-meta">
                  <strong>
                    {v.name}
                    {v.demo ? <span className="voice-demo-badge">Demo</span> : null}
                  </strong>
                  <small>
                    {v.gender} · {v.style} · {v.accent}
                  </small>
                </div>
                <input
                  type="radio"
                  name="showcase-voice"
                  checked={selected}
                  readOnly
                  tabIndex={-1}
                  aria-hidden
                />
              </div>
              <p className="voice-card-desc">{v.description}</p>
              <div className="voice-card-actions">
                <button
                  type="button"
                  className="voice-preview-btn voice-preview-wide"
                  disabled={previewing === v.id}
                  onClick={(e) => {
                    e.stopPropagation();
                    preview(v);
                  }}
                >
                  <Play size={12} fill="currentColor" />
                  {previewing === v.id ? "Playing…" : "Preview"}
                </button>
                {selected && (
                  <span className="voice-selected-label">
                    <Mic2 size={12} /> Selected
                  </span>
                )}
              </div>
            </article>
          );
        })}
      </div>

      <p className="voices-footnote">
        <Sparkles size={12} />
        Demo voices preview with your browser TTS. Connect ElevenLabs in{" "}
        <code>.env</code> for production neural voices.
      </p>
    </div>
  );
}
