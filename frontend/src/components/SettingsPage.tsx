import { useEffect, useState } from "react";
import { Activity, Hand, Move, Check, RefreshCw } from "lucide-react";
import { fetchVocabulary, reloadModels, type Vocabulary } from "../lib/api";

export type RecognitionMode = "AUTO" | "STATIC" | "DYNAMIC";

const MODES: {
  id: RecognitionMode;
  title: string;
  subtitle: string;
  detail: string;
  icon: React.ReactNode;
}[] = [
  {
    id: "STATIC",
    title: "Static",
    subtitle: "Held poses (recommended)",
    detail:
      "Uses your RandomForest model from collect_data.py / train_model.py. Best for STOP, EAT, DRINK, MORE, and other held signs.",
    icon: <Hand size={22} strokeWidth={2} />,
  },
  {
    id: "AUTO",
    title: "Auto",
    subtitle: "Static + motion",
    detail:
      "Combines both models. Motion LSTM only knows HELLO/HELP/YES/… — new pose-only signs use the static model.",
    icon: <Activity size={22} strokeWidth={2} />,
  },
  {
    id: "DYNAMIC",
    title: "Dynamic",
    subtitle: "Motion signs only",
    detail:
      "Only the LSTM from collect_dynamic.py. Limited to motion vocabulary until you retrain it.",
    icon: <Move size={22} strokeWidth={2} />,
  },
];

type Props = {
  mode: RecognitionMode;
  onModeChange: (mode: RecognitionMode) => void;
};

export function SettingsPage({ mode, onModeChange }: Props) {
  const [vocab, setVocab] = useState<Vocabulary | null>(null);
  const [vocabError, setVocabError] = useState<string | null>(null);
  const [reloading, setReloading] = useState(false);

  const loadVocab = () => {
    fetchVocabulary()
      .then((v) => {
        setVocab(v);
        setVocabError(null);
      })
      .catch(() => {
        setVocab(null);
        setVocabError("Could not load vocabulary — is the API running?");
      });
  };

  useEffect(() => {
    loadVocab();
  }, []);

  const onReload = async () => {
    setReloading(true);
    try {
      const v = await reloadModels();
      setVocab(v);
      setVocabError(null);
    } catch {
      setVocabError("Reload failed — restart python run_web.py after training.");
    } finally {
      setReloading(false);
    }
  };

  return (
    <div className="settings-page">
      <header className="page-header">
        <div className="page-header-text">
          <h1>Settings</h1>
          <p>Choose how signs are recognized during live translate</p>
        </div>
      </header>

      <section className="settings-section">
        <h2 className="settings-section-title">Recognition mode</h2>
        <p className="settings-section-desc">
          After collecting new signs, run <code>python train_model.py</code>, then
          reload models below (or restart the API). Use <strong>Static</strong> for
          pose signs like STOP / EAT / DRINK.
        </p>

        <div className="mode-grid" role="radiogroup" aria-label="Recognition mode">
          {MODES.map((m) => {
            const selected = mode === m.id;
            return (
              <button
                key={m.id}
                type="button"
                role="radio"
                aria-checked={selected}
                className={`mode-card ${selected ? "selected" : ""}`}
                onClick={() => onModeChange(m.id)}
              >
                <div className="mode-card-top">
                  <span className="mode-icon">{m.icon}</span>
                  {selected && (
                    <span className="mode-check" aria-hidden>
                      <Check size={16} strokeWidth={2.5} />
                    </span>
                  )}
                </div>
                <strong className="mode-title">{m.title}</strong>
                <span className="mode-subtitle">{m.subtitle}</span>
                <p className="mode-detail">{m.detail}</p>
              </button>
            );
          })}
        </div>

        <p className="settings-current">
          Current mode: <strong>{mode}</strong>
        </p>
      </section>

      <section className="settings-section">
        <div className="settings-vocab-header">
          <div>
            <h2 className="settings-section-title">Trained signs (loaded model)</h2>
            <p className="settings-section-desc">
              What the API currently has in memory — not a wishlist.{" "}
              <strong>WATER</strong> only appears after you collect + train it
              (you have <strong>DRINK</strong> instead).
            </p>
          </div>
          <button
            type="button"
            className="btn-text-sm vocab-reload"
            onClick={onReload}
            disabled={reloading}
          >
            <RefreshCw size={14} className={reloading ? "spin" : undefined} />
            {reloading ? "Reloading…" : "Reload models"}
          </button>
        </div>

        {vocabError && <p className="settings-vocab-error">{vocabError}</p>}

        {vocab && (
          <div className="vocab-blocks">
            <div className="vocab-block card">
              <h3>Static ({vocab.static_signs.length})</h3>
              <div className="vocab-chips">
                {vocab.static_signs.map((s) => (
                  <span key={s} className="vocab-chip">
                    {s.replace(/_/g, " ")}
                  </span>
                ))}
              </div>
            </div>
            <div className="vocab-block card">
              <h3>Dynamic / motion ({vocab.dynamic_signs.length})</h3>
              <div className="vocab-chips">
                {vocab.dynamic_signs.length ? (
                  vocab.dynamic_signs.map((s) => (
                    <span key={s} className="vocab-chip muted">
                      {s.replace(/_/g, " ")}
                    </span>
                  ))
                ) : (
                  <span className="vocab-empty">No dynamic model loaded</span>
                )}
              </div>
            </div>
          </div>
        )}
      </section>
    </div>
  );
}
