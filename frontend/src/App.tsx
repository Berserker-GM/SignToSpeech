import { useCallback, useEffect, useRef, useState } from "react";
import { Sidebar, type Page } from "./components/Sidebar";
import { LiveTranslate } from "./components/LiveTranslate";
import { SpeechPanel } from "./components/SpeechPanel";
import { PlaceholderPage } from "./components/PlaceholderPage";
import { HistoryPage } from "./components/HistoryPage";
import { VoicesPage } from "./components/VoicesPage";
import {
  SettingsPage,
  type RecognitionMode,
} from "./components/SettingsPage";
import { clearSession } from "./lib/api";

// v2 default STATIC — AUTO was overriding new pose signs with the old LSTM vocab
const MODE_KEY = "sts_recognition_mode_v2";

function loadMode(): RecognitionMode {
  const saved = localStorage.getItem(MODE_KEY);
  if (saved === "AUTO" || saved === "STATIC" || saved === "DYNAMIC") return saved;
  return "STATIC";
}

export default function App() {
  const [page, setPage] = useState<Page>("home");
  const [lastSpoken, setLastSpoken] = useState<string | null>(null);
  const [sentence, setSentence] = useState<string[]>([]);
  const [recognitionMode, setRecognitionMode] = useState<RecognitionMode>(loadMode);
  const [historyRefresh, setHistoryRefresh] = useState(0);
  const clearLiveRef = useRef<() => void>(() => {});

  useEffect(() => {
    localStorage.setItem(MODE_KEY, recognitionMode);
  }, [recognitionMode]);

  const onWordSpoken = (word: string) => {
    setLastSpoken(word);
    setTimeout(() => setLastSpoken(null), 2500);
  };

  const clearAll = useCallback(() => {
    clearLiveRef.current();
    setSentence([]);
    setLastSpoken(null);
    clearSession().catch(() => {});
  }, []);

  const showLive = page === "live" || page === "home";

  return (
    <div className="app-shell">
      <Sidebar page={page} onNavigate={setPage} />

      <div className="main-area">
        {showLive && (
          <LiveTranslate
            onWordSpoken={onWordSpoken}
            onSentenceChange={setSentence}
            onClear={clearAll}
            recognitionMode={recognitionMode}
            onSavedToHistory={() => setHistoryRefresh((n) => n + 1)}
            registerClear={(fn) => {
              clearLiveRef.current = fn;
            }}
          />
        )}
        {page === "history" && <HistoryPage refreshKey={historyRefresh} />}
        {page === "voices" && <VoicesPage />}
        {page === "settings" && (
          <SettingsPage
            mode={recognitionMode}
            onModeChange={setRecognitionMode}
          />
        )}
        {page === "help" && (
          <PlaceholderPage
            title="Help"
            description="Hold each sign steady ~1 second to add it to your sentence. Press Speak Now for the full corrected sentence. Use Save in history to keep conversations under History."
          />
        )}
        {page === "about" && (
          <PlaceholderPage
            title="About"
            description="Sign Language to Speech — bridging communication in hospitals, transit, and public spaces."
          />
        )}

        {showLive && (
          <SpeechPanel
            sentence={sentence}
            autoSpeakWord={lastSpoken}
            onClear={clearAll}
          />
        )}
      </div>
    </div>
  );
}
