import { useCallback, useEffect, useRef, useState } from "react";
import {
  ActivityIndicator,
  Pressable,
  StyleSheet,
  Text,
  View,
} from "react-native";
import { useKeepAwake } from "expo-keep-awake";
import { HandTrackerWebView } from "./HandTrackerWebView";
import { useLiveSession } from "../hooks/useLiveSession";
import { correctSentence, clearSession } from "../lib/api";
import { formatDisplay, formatGloss } from "../lib/format";
import { playSpeakResult, stopSpeech } from "../lib/tts";
import type { Voice } from "../lib/api";
import type { RecognitionMode } from "./SettingsScreen";

type Props = {
  apiBase: string;
  voice: Voice | undefined;
  recognitionMode: RecognitionMode;
  onModeChange: (mode: RecognitionMode) => void;
  onOpenSettings: () => void;
};

export function LiveScreen({
  apiBase,
  voice,
  recognitionMode,
  onModeChange,
  onOpenSettings,
}: Props) {
  useKeepAwake();

  const [paused, setPaused] = useState(false);
  const [trackerReady, setTrackerReady] = useState(false);
  const [cameraStatus, setCameraStatus] = useState<string>("starting");
  const [translated, setTranslated] = useState("—");
  const [rawSentence, setRawSentence] = useState("");
  const [grammarActive, setGrammarActive] = useState(false);
  const [speaking, setSpeaking] = useState(false);
  const [notice, setNotice] = useState<string | null>(null);
  const lastConfirmed = useRef<string | null>(null);

  const { connected, frame, sentence, sendLandmarks, clearSentence } =
    useLiveSession(apiBase, !paused, recognitionMode);

  const onLandmarks = useCallback(
    (landmarks: number[] | null, handDetected: boolean) => {
      if (paused) return;
      if (landmarks && handDetected) sendLandmarks(landmarks);
      else sendLandmarks(null);
    },
    [paused, sendLandmarks]
  );

  useEffect(() => {
    if (frame.confirmed && frame.confirmed !== lastConfirmed.current) {
      lastConfirmed.current = frame.confirmed;
      const word = frame.confirmed.replace(/_/g, " ");
      (async () => {
        try {
          setSpeaking(true);
          const r = await playSpeakResult(apiBase, word, voice, 0.9);
          if (r.warning) setNotice(r.warning);
        } catch {
          /* ignore */
        } finally {
          setSpeaking(false);
        }
      })();
    }
  }, [frame.confirmed, apiBase, voice]);

  useEffect(() => {
    if (!sentence.length) {
      setTranslated("—");
      setRawSentence("");
      return;
    }
    const raw = sentence.join(" ");
    setRawSentence(raw);
    const t = setTimeout(async () => {
      try {
        const r = await correctSentence(apiBase, sentence);
        setTranslated(r.corrected);
        setGrammarActive(r.grammar_active);
        if (!r.grammar_active && r.error) setNotice(r.error);
      } catch {
        setTranslated(sentence.map((w) => formatDisplay(w)).join(" "));
        setGrammarActive(false);
        setNotice("Grammar request failed");
      }
    }, 400);
    return () => clearTimeout(t);
  }, [sentence, apiBase]);

  const clearAll = async () => {
    stopSpeech();
    clearSentence();
    lastConfirmed.current = null;
    setTranslated("—");
    setRawSentence("");
    try {
      await clearSession(apiBase);
    } catch {
      /* ignore */
    }
  };

  const speakNow = async () => {
    if (!sentence.length) return;
    setSpeaking(true);
    setNotice(null);
    try {
      let text = translated;
      if (text === "—" || !text) {
        const r = await correctSentence(apiBase, sentence);
        text = r.corrected;
        setTranslated(text);
        setGrammarActive(r.grammar_active);
      }
      const r = await playSpeakResult(apiBase, text, voice, 0.9);
      if (r.warning) setNotice(r.warning);
    } catch (e) {
      setNotice((e as Error).message || "Speech failed");
    } finally {
      setSpeaking(false);
    }
  };

  const cycleMode = () => {
    const modes: RecognitionMode[] = ["AUTO", "STATIC", "DYNAMIC"];
    const current = (frame.mode as RecognitionMode) || recognitionMode;
    const next = modes[(modes.indexOf(current) + 1) % modes.length];
    onModeChange(next);
  };

  const detected = frame.final_pred;
  const conf = frame.final_conf ?? 0;
  const hasSign = frame.hand_detected && detected;
  const hold = frame.hold_progress ?? 0;

  return (
    <View style={styles.root}>
      <View style={styles.header}>
        <View style={{ flex: 1 }}>
          <Text style={styles.title}>Live Sign Detection</Text>
          <Text style={styles.sub}>
            {connected
              ? "Connected to STS models"
              : "Not connected — check server URL in Settings"}
          </Text>
        </View>
        <Pressable style={styles.chip} onPress={onOpenSettings}>
          <Text style={styles.chipText}>Settings</Text>
        </Pressable>
      </View>

      <View style={styles.cameraCard}>
        <HandTrackerWebView
          paused={paused}
          onLandmarks={onLandmarks}
          onReady={() => setTrackerReady(true)}
          onCameraStatus={(s, msg) => {
            setCameraStatus(s);
            if (s === "denied") setNotice(msg || "Camera denied");
          }}
          onError={(m) => setNotice(m)}
        />
        {!trackerReady && (
          <View style={styles.loading}>
            <ActivityIndicator color="#fff" />
            <Text style={styles.loadingText}>Loading hand tracker…</Text>
          </View>
        )}
        {hold > 0 && (
          <View style={styles.holdBadge}>
            <Text style={styles.holdText}>{Math.round(hold * 100)}%</Text>
          </View>
        )}
        <View style={styles.camControls}>
          <Pressable
            style={[styles.ctrlBtn, styles.ctrlPrimary]}
            onPress={() => setPaused((p) => !p)}
          >
            <Text style={styles.ctrlPrimaryText}>
              {paused ? "Resume" : "Pause"}
            </Text>
          </Pressable>
        </View>
      </View>

      <View style={styles.row}>
        <View style={styles.card}>
          <Text style={styles.cardLabel}>Detected</Text>
          <Text style={[styles.sign, !hasSign && styles.muted]}>
            {hasSign ? formatDisplay(detected) : "—"}
          </Text>
          {hasSign && conf > 0 && (
            <Text style={styles.conf}>{Math.round(conf * 100)}% · {frame.final_source || ""}</Text>
          )}
          <Text style={styles.gloss}>Gloss: {hasSign ? formatGloss(detected) : "—"}</Text>
        </View>
        <View style={styles.card}>
          <Text style={styles.cardLabel}>Sentence</Text>
          <Text style={[styles.sentence, translated === "—" && styles.muted]} numberOfLines={4}>
            {translated}
          </Text>
          {grammarActive && translated !== "—" && (
            <Text style={styles.tag}>Grammar corrected</Text>
          )}
          {!!rawSentence && (
            <Text style={styles.raw} numberOfLines={2}>
              Raw: {rawSentence}
            </Text>
          )}
        </View>
      </View>

      {notice && <Text style={styles.notice}>{notice}</Text>}

      <View style={styles.actions}>
        <Pressable
          style={[styles.speakBtn, (!sentence.length || speaking) && styles.disabled]}
          disabled={!sentence.length || speaking}
          onPress={speakNow}
        >
          <Text style={styles.speakText}>
            {speaking ? "Speaking…" : "Speak Now"}
          </Text>
        </Pressable>
        <Pressable style={styles.secondaryBtn} onPress={cycleMode}>
          <Text style={styles.secondaryText}>
            {frame.mode || recognitionMode}
          </Text>
        </Pressable>
        <Pressable style={styles.secondaryBtn} onPress={clearAll}>
          <Text style={styles.secondaryText}>Clear</Text>
        </Pressable>
      </View>

      <Text style={styles.footer}>
        Camera: {cameraStatus}
        {" · "}
        Hold a sign ~1s to add a word
      </Text>
    </View>
  );
}

const styles = StyleSheet.create({
  root: {
    flex: 1,
    backgroundColor: "#f5f7fb",
    paddingHorizontal: 16,
    paddingTop: 8,
    paddingBottom: 12,
  },
  header: {
    flexDirection: "row",
    alignItems: "flex-start",
    gap: 10,
    marginBottom: 12,
  },
  title: {
    fontSize: 20,
    fontWeight: "700",
    color: "#111827",
    letterSpacing: -0.3,
  },
  sub: {
    marginTop: 4,
    fontSize: 12,
    color: "#6b7280",
  },
  chip: {
    backgroundColor: "#ede9fe",
    paddingHorizontal: 12,
    paddingVertical: 8,
    borderRadius: 10,
  },
  chipText: {
    color: "#7c3aed",
    fontWeight: "600",
    fontSize: 13,
  },
  cameraCard: {
    height: 320,
    borderRadius: 16,
    overflow: "hidden",
    backgroundColor: "#0f1115",
  },
  loading: {
    position: "absolute",
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
    alignItems: "center",
    justifyContent: "center",
    backgroundColor: "rgba(0,0,0,0.55)",
    gap: 10,
  },
  loadingText: { color: "#fff", fontSize: 13 },
  holdBadge: {
    position: "absolute",
    top: 12,
    right: 12,
    backgroundColor: "rgba(16,185,129,0.9)",
    borderRadius: 20,
    paddingHorizontal: 10,
    paddingVertical: 6,
  },
  holdText: { color: "#fff", fontWeight: "700", fontSize: 12 },
  camControls: {
    position: "absolute",
    bottom: 12,
    left: 0,
    right: 0,
    alignItems: "center",
  },
  ctrlBtn: {
    paddingHorizontal: 18,
    paddingVertical: 10,
    borderRadius: 999,
  },
  ctrlPrimary: { backgroundColor: "#7c3aed" },
  ctrlPrimaryText: { color: "#fff", fontWeight: "700", fontSize: 14 },
  row: {
    flexDirection: "row",
    gap: 10,
    marginTop: 12,
  },
  card: {
    flex: 1,
    backgroundColor: "#fff",
    borderRadius: 14,
    padding: 14,
    borderWidth: 1,
    borderColor: "#eef0f4",
    minHeight: 120,
  },
  cardLabel: {
    fontSize: 12,
    fontWeight: "600",
    color: "#9ca3af",
    marginBottom: 8,
  },
  sign: {
    fontSize: 22,
    fontWeight: "700",
    color: "#7c3aed",
  },
  sentence: {
    fontSize: 15,
    fontWeight: "600",
    color: "#111827",
    lineHeight: 21,
  },
  muted: { color: "#d1d5db" },
  conf: {
    marginTop: 6,
    fontSize: 12,
    color: "#047857",
    fontWeight: "600",
  },
  gloss: {
    marginTop: 8,
    fontSize: 12,
    color: "#9ca3af",
  },
  tag: {
    marginTop: 8,
    alignSelf: "flex-start",
    fontSize: 11,
    fontWeight: "600",
    color: "#047857",
    backgroundColor: "#d1fae5",
    paddingHorizontal: 8,
    paddingVertical: 4,
    borderRadius: 999,
  },
  raw: {
    marginTop: 8,
    fontSize: 11,
    color: "#9ca3af",
  },
  notice: {
    marginTop: 10,
    fontSize: 12,
    color: "#b45309",
  },
  actions: {
    flexDirection: "row",
    gap: 8,
    marginTop: 14,
  },
  speakBtn: {
    flex: 1,
    backgroundColor: "#7c3aed",
    borderRadius: 12,
    paddingVertical: 14,
    alignItems: "center",
  },
  speakText: { color: "#fff", fontWeight: "700", fontSize: 15 },
  secondaryBtn: {
    paddingHorizontal: 14,
    borderRadius: 12,
    borderWidth: 1,
    borderColor: "#e5e7eb",
    backgroundColor: "#fff",
    justifyContent: "center",
  },
  secondaryText: {
    color: "#374151",
    fontWeight: "600",
    fontSize: 13,
  },
  disabled: { opacity: 0.45 },
  footer: {
    marginTop: 12,
    fontSize: 11,
    color: "#9ca3af",
    textAlign: "center",
  },
});
