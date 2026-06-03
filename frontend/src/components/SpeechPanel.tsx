import { useEffect, useRef, useState } from "react";
import { Volume2, VolumeX, Square, Sparkles, AlertCircle, Trash2 } from "lucide-react";
import type { Voice } from "../lib/api";
import { audioSrcFromResult, correctSentence, fetchVoices, speakText } from "../lib/api";
import { speakBrowser, stopBrowserSpeech } from "../lib/browserTts";

type Props = {
  sentence: string[];
  autoSpeakWord?: string | null;
  onClear: () => void;
};

export function SpeechPanel({ sentence, autoSpeakWord, onClear }: Props) {
  const [voices, setVoices] = useState<Voice[]>([]);
  const [voiceId, setVoiceId] = useState("browser");
  const [displayText, setDisplayText] = useState("—");
  const [correctedText, setCorrectedText] = useState("");
  const [grammarActive, setGrammarActive] = useState(true);
  const [volume, setVolume] = useState(0.8);
  const [playing, setPlaying] = useState(false);
  const [loading, setLoading] = useState(false);
  const [ttsNotice, setTtsNotice] = useState<string | null>(null);
  const audioRef = useRef<HTMLAudioElement | null>(null);

  useEffect(() => {
    fetchVoices().then(({ voices: v, defaultId }) => {
      setVoices(v.slice(0, 2));
      setVoiceId(defaultId);
    });
    if ("speechSynthesis" in window) {
      speechSynthesis.getVoices();
      speechSynthesis.onvoiceschanged = () => speechSynthesis.getVoices();
    }
  }, []);

  useEffect(() => {
    if (!sentence.length) {
      setDisplayText("—");
      setCorrectedText("");
      return;
    }
    const raw = sentence.join(" ");
    setDisplayText(raw);

    const t = setTimeout(async () => {
      try {
        const r = await correctSentence(sentence);
        setCorrectedText(r.corrected);
        setGrammarActive(r.grammar_active);
      } catch {
        setCorrectedText("");
      }
    }, 400);
    return () => clearTimeout(t);
  }, [sentence]);

  useEffect(() => {
    if (!sentence.length) {
      setCorrectedText("");
      setDisplayText("—");
    }
  }, [sentence.length]);

  const runSpeech = async (text: string, useGrammar: boolean) => {
    if (!text.trim() || text === "—") return;
    setLoading(true);
    setTtsNotice(null);
    stopBrowserSpeech();
    audioRef.current?.pause();

    try {
      let toSpeak = text;
      if (useGrammar && sentence.length > 0) {
        const result = await correctSentence(sentence);
        toSpeak = result.corrected;
        setCorrectedText(toSpeak);
        setGrammarActive(result.grammar_active);
      } else if (correctedText) {
        toSpeak = correctedText;
      }

      const selected = voices.find((v) => v.id === voiceId);
      const useBrowserVoice =
        voiceId === "browser" || selected?.provider === "browser";

      if (useBrowserVoice) {
        setPlaying(true);
        await speakBrowser(toSpeak, volume);
        setPlaying(false);
        return;
      }

      const result = await speakText(toSpeak, voiceId);
      if (result.warning) setTtsNotice(result.warning);

      if (result.provider === "browser") {
        setPlaying(true);
        await speakBrowser(result.text || toSpeak, volume);
        setPlaying(false);
        return;
      }

      const src = audioSrcFromResult(result);
      if (!src) throw new Error("No audio returned");

      const audio = new Audio(src);
      audio.volume = volume;
      audioRef.current = audio;
      setPlaying(true);
      audio.onended = () => setPlaying(false);
      await audio.play();
    } catch (e) {
      console.error(e);
      try {
        setTtsNotice("Using browser voice (ElevenLabs unavailable).");
        setPlaying(true);
        await speakBrowser(correctedText || text, volume);
        setPlaying(false);
      } catch {
        alert((e as Error).message || "Speech failed");
      }
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (!autoSpeakWord) return;
    runSpeech(autoSpeakWord.replace(/_/g, " "), false);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [autoSpeakWord]);

  const stop = () => {
    stopBrowserSpeech();
    audioRef.current?.pause();
    setPlaying(false);
  };

  const previewVoice = async (v: Voice) => {
    const sample = `Hello, I am ${v.name}.`;
    if (v.id === "browser" || v.provider === "browser") {
      await speakBrowser(sample, volume);
      return;
    }
    try {
      const result = await speakText(sample, v.id);
      if (result.provider === "browser") {
        await speakBrowser(sample, volume);
        return;
      }
      const src = audioSrcFromResult(result);
      if (src) {
        const a = new Audio(src);
        a.volume = volume;
        await a.play();
      }
    } catch {
      await speakBrowser(sample, volume);
    }
  };

  const speakLabel = correctedText || displayText;
  const hasContent = sentence.length > 0;

  return (
    <aside className="right-panel">
      <div className="panel-card speech-panel-card">
        <div className="panel-card-head">
          <h3 className="panel-card-title">Speech Output</h3>
          {hasContent && (
            <button
              type="button"
              className="btn-clear-data"
              onClick={onClear}
              title="Clear all signs and text"
            >
              <Trash2 size={15} />
              Clear
            </button>
          )}
        </div>

        <div className="speech-output-box">
          {hasContent ? (
            <>
              {correctedText && (
                <p className="speech-output-corrected">{correctedText}</p>
              )}
              <p className="speech-output-raw">Raw: {displayText}</p>
            </>
          ) : (
            <p className="speech-output-placeholder">—</p>
          )}
        </div>

        {ttsNotice && (
          <p className="tts-notice">
            <AlertCircle size={14} style={{ flexShrink: 0 }} />
            {ttsNotice}
          </p>
        )}

        <div className={`waveform ${playing ? "playing" : ""}`}>
          {Array.from({ length: 20 }).map((_, i) => (
            <span key={i} />
          ))}
        </div>

        <div className="speak-row">
          <button
            type="button"
            className="btn-speak"
            disabled={loading || !hasContent}
            onClick={() => runSpeech(speakLabel, true)}
          >
            <Volume2 size={18} strokeWidth={2.5} />
            {loading ? "Processing…" : "Speak Now"}
          </button>
          <button type="button" className="btn-stop" onClick={stop} aria-label="Stop">
            <Square size={16} fill="currentColor" />
          </button>
        </div>

        <div className="volume-row">
          {volume < 0.05 ? <VolumeX size={18} /> : <Volume2 size={18} />}
          <input
            type="range"
            min={0}
            max={1}
            step={0.05}
            value={volume}
            aria-label="Volume"
            onChange={(e) => {
              const v = Number(e.target.value);
              setVolume(v);
              if (audioRef.current) audioRef.current.volume = v;
            }}
          />
        </div>
      </div>

      <div className="panel-card">
        <h3 className="panel-card-title">Voice</h3>
        <p className="voice-section-sub">System default or your ElevenLabs voice</p>
        <div className="voice-list voice-list-compact">
          {voices.map((v) => (
            <div
              key={v.id}
              className={`voice-item ${voiceId === v.id ? "selected" : ""}`}
              onClick={() => setVoiceId(v.id)}
              role="button"
              tabIndex={0}
            >
              <input type="radio" name="voice" checked={voiceId === v.id} readOnly />
              <div className="voice-avatar">{v.name[0]}</div>
              <div className="voice-meta">
                <strong>{v.name}</strong>
                <small>{v.gender}</small>
              </div>
              <button
                type="button"
                className="voice-preview-btn"
                onClick={(e) => {
                  e.stopPropagation();
                  previewVoice(v);
                }}
                aria-label={`Preview ${v.name}`}
              >
                ▶
              </button>
            </div>
          ))}
        </div>
        <p className="eleven-footer">
          <Sparkles size={12} />
          {voices.find((v) => v.provider === "elevenlabs")
            ? `ElevenLabs: ${voices.find((v) => v.provider === "elevenlabs")!.name}`
            : "Add ELEVENLABS_VOICE_ID in .env"}
        </p>
      </div>
    </aside>
  );
}
