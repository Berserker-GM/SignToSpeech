import { useCallback, useRef, useState } from "react";
import { Sidebar, type Page } from "./components/Sidebar";
import { LiveTranslate } from "./components/LiveTranslate";
import { SpeechPanel } from "./components/SpeechPanel";
import { PlaceholderPage } from "./components/PlaceholderPage";
import { clearSession } from "./lib/api";

export default function App() {
  const [page, setPage] = useState<Page>("home");
  const [lastSpoken, setLastSpoken] = useState<string | null>(null);
  const [sentence, setSentence] = useState<string[]>([]);
  const clearLiveRef = useRef<() => void>(() => {});

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
            registerClear={(fn) => {
              clearLiveRef.current = fn;
            }}
          />
        )}
        {page === "history" && (
          <PlaceholderPage
            title="History"
            description="Past translated sentences will appear here in a future update."
          />
        )}
        {page === "voices" && (
          <PlaceholderPage
            title="Voices"
            description="Use the voice panel on Live Translate to pick ElevenLabs voices."
          />
        )}
        {page === "settings" && (
          <PlaceholderPage
            title="Settings"
            description="Thresholds, modes (AUTO / STATIC / DYNAMIC), and API keys via .env."
          />
        )}
        {page === "help" && (
          <PlaceholderPage
            title="Help"
            description="Hold each sign steady ~1 second to add it to your sentence. Press Speak Now for the full corrected sentence."
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
