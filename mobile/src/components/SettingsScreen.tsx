import { useEffect, useState } from "react";
import {
  ActivityIndicator,
  Pressable,
  StyleSheet,
  Text,
  TextInput,
  View,
} from "react-native";
import {
  checkHealth,
  fetchVoices,
  type Voice,
} from "../lib/api";
import { getDefaultApiBase, setApiBaseUrl } from "../lib/config";

export type RecognitionMode = "AUTO" | "STATIC" | "DYNAMIC";

const MODES: { id: RecognitionMode; title: string; desc: string }[] = [
  {
    id: "AUTO",
    title: "Auto",
    desc: "Static + motion — picks the more confident model",
  },
  {
    id: "STATIC",
    title: "Static",
    desc: "Letters & held poses only (RandomForest)",
  },
  {
    id: "DYNAMIC",
    title: "Dynamic",
    desc: "Motion signs only (LSTM)",
  },
];

type Props = {
  apiBase: string;
  voiceId: string;
  recognitionMode: RecognitionMode;
  onModeChange: (mode: RecognitionMode) => void;
  onSaved: (base: string, voices: Voice[], defaultId: string) => void;
  onBack: () => void;
  onSelectVoice: (id: string) => void;
  voices: Voice[];
};

export function SettingsScreen({
  apiBase,
  voiceId,
  recognitionMode,
  onModeChange,
  onSaved,
  onBack,
  onSelectVoice,
  voices,
}: Props) {
  const suggested = getDefaultApiBase();
  const [url, setUrl] = useState(apiBase || suggested);
  const [checking, setChecking] = useState(false);
  const [status, setStatus] = useState<string | null>(null);

  useEffect(() => {
    setUrl(apiBase || suggested);
  }, [apiBase, suggested]);

  const saveAndTest = async () => {
    setChecking(true);
    setStatus(null);
    try {
      const base = await setApiBaseUrl(url);
      const ok = await checkHealth(base);
      if (!ok) {
        setStatus(
          `Cannot reach API at ${base}. On your PC run: python run_web.py — phone/PC must be on the same Wi‑Fi. Suggested: ${suggested}`
        );
        setChecking(false);
        return;
      }
      const { voices: v, defaultId } = await fetchVoices(base);
      onSaved(base, v, defaultId);
      setStatus("Connected — models are ready.");
    } catch (e) {
      setStatus((e as Error).message || "Connection failed");
    } finally {
      setChecking(false);
    }
  };

  return (
    <View style={styles.root}>
      <View style={styles.header}>
        <Pressable onPress={onBack}>
          <Text style={styles.back}>← Back</Text>
        </Pressable>
        <Text style={styles.title}>Settings</Text>
      </View>

      <Text style={styles.label}>STS API server URL</Text>
      <Text style={styles.hint}>
        Web: use http://127.0.0.1:8000. Phone: same Wi‑Fi as your PC. Suggested:{" "}
        {suggested}
      </Text>
      <TextInput
        style={styles.input}
        value={url}
        onChangeText={setUrl}
        autoCapitalize="none"
        autoCorrect={false}
        keyboardType="url"
        placeholder={suggested}
        placeholderTextColor="#9ca3af"
      />

      <Pressable style={styles.useSuggested} onPress={() => setUrl(suggested)}>
        <Text style={styles.useSuggestedText}>Use suggested URL</Text>
      </Pressable>

      <Pressable
        style={[styles.btn, checking && styles.disabled]}
        onPress={saveAndTest}
        disabled={checking}
      >
        {checking ? (
          <ActivityIndicator color="#fff" />
        ) : (
          <Text style={styles.btnText}>Save & Test Connection</Text>
        )}
      </Pressable>

      {status && <Text style={styles.status}>{status}</Text>}

      <Text style={[styles.label, { marginTop: 28 }]}>Recognition mode</Text>
      <Text style={styles.hint}>
        Choose Auto, Static, or Dynamic. Applies on Live Translate.
      </Text>
      {MODES.map((m) => (
        <Pressable
          key={m.id}
          style={[styles.voiceRow, recognitionMode === m.id && styles.modeSelected]}
          onPress={() => onModeChange(m.id)}
        >
          <View style={{ flex: 1 }}>
            <Text style={styles.voiceName}>{m.title}</Text>
            <Text style={styles.voiceMeta}>{m.desc}</Text>
          </View>
          <Text style={styles.radio}>{recognitionMode === m.id ? "●" : "○"}</Text>
        </Pressable>
      ))}

      <Text style={[styles.label, { marginTop: 28 }]}>Voice</Text>
      <Text style={styles.hint}>
        Device voice works offline. ElevenLabs appears if the API key is set on the server.
      </Text>
      {voices.length === 0 ? (
        <Text style={styles.empty}>Connect to the API to load voices.</Text>
      ) : (
        voices.map((v) => (
          <Pressable
            key={v.id}
            style={[styles.voiceRow, voiceId === v.id && styles.voiceSelected]}
            onPress={() => onSelectVoice(v.id)}
          >
            <View style={{ flex: 1 }}>
              <Text style={styles.voiceName}>{v.name}</Text>
              <Text style={styles.voiceMeta}>{v.gender}</Text>
            </View>
            <Text style={styles.radio}>{voiceId === v.id ? "●" : "○"}</Text>
          </Pressable>
        ))
      )}
    </View>
  );
}

const styles = StyleSheet.create({
  root: {
    flex: 1,
    backgroundColor: "#f5f7fb",
    paddingHorizontal: 16,
    paddingTop: 8,
  },
  header: { marginBottom: 20 },
  back: { color: "#7c3aed", fontWeight: "600", marginBottom: 8 },
  title: {
    fontSize: 22,
    fontWeight: "700",
    color: "#111827",
  },
  label: {
    fontSize: 14,
    fontWeight: "700",
    color: "#111827",
    marginBottom: 6,
  },
  hint: {
    fontSize: 12,
    color: "#6b7280",
    marginBottom: 10,
    lineHeight: 17,
  },
  input: {
    backgroundColor: "#fff",
    borderWidth: 1,
    borderColor: "#e5e7eb",
    borderRadius: 12,
    paddingHorizontal: 14,
    paddingVertical: 12,
    fontSize: 15,
    color: "#111827",
    marginBottom: 8,
  },
  useSuggested: {
    alignSelf: "flex-start",
    marginBottom: 12,
    paddingVertical: 4,
  },
  useSuggestedText: {
    color: "#7c3aed",
    fontWeight: "600",
    fontSize: 13,
  },
  btn: {
    backgroundColor: "#7c3aed",
    borderRadius: 12,
    paddingVertical: 14,
    alignItems: "center",
  },
  btnText: { color: "#fff", fontWeight: "700", fontSize: 15 },
  disabled: { opacity: 0.6 },
  status: {
    marginTop: 12,
    fontSize: 13,
    color: "#374151",
    lineHeight: 18,
  },
  empty: { color: "#9ca3af", fontSize: 13 },
  voiceRow: {
    flexDirection: "row",
    alignItems: "center",
    backgroundColor: "#fff",
    borderRadius: 12,
    borderWidth: 1,
    borderColor: "#eef0f4",
    padding: 14,
    marginBottom: 8,
  },
  voiceSelected: {
    borderColor: "#c4b5fd",
    backgroundColor: "#f5f3ff",
  },
  modeSelected: {
    borderColor: "#5eead4",
    backgroundColor: "#ccfbf1",
  },
  voiceName: { fontWeight: "700", color: "#111827", fontSize: 14 },
  voiceMeta: { marginTop: 2, color: "#9ca3af", fontSize: 12 },
  radio: { color: "#7c3aed", fontSize: 18, marginLeft: 8 },
});
