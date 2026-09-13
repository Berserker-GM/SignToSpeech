import { useEffect, useState } from "react";
import { ActivityIndicator, StyleSheet, Text, View } from "react-native";
import { StatusBar } from "expo-status-bar";
import { SafeAreaProvider, SafeAreaView } from "react-native-safe-area-context";
import { LiveScreen } from "./src/components/LiveScreen";
import {
  SettingsScreen,
  type RecognitionMode,
} from "./src/components/SettingsScreen";
import { checkHealth, fetchVoices, type Voice } from "./src/lib/api";
import { getApiBaseUrl, getDefaultApiBase } from "./src/lib/config";
import AsyncStorage from "@react-native-async-storage/async-storage";

type Page = "live" | "settings";
const MODE_KEY = "sts_recognition_mode";

export default function App() {
  const [page, setPage] = useState<Page>("live");
  const [booting, setBooting] = useState(true);
  const [apiBase, setApiBase] = useState(getDefaultApiBase);
  const [voices, setVoices] = useState<Voice[]>([]);
  const [voiceId, setVoiceId] = useState("browser");
  const [bootNote, setBootNote] = useState<string | null>(null);
  const [recognitionMode, setRecognitionMode] =
    useState<RecognitionMode>("AUTO");

  useEffect(() => {
    (async () => {
      const savedMode = await AsyncStorage.getItem(MODE_KEY);
      if (
        savedMode === "AUTO" ||
        savedMode === "STATIC" ||
        savedMode === "DYNAMIC"
      ) {
        setRecognitionMode(savedMode);
      }
      const base = await getApiBaseUrl();
      setApiBase(base);
      const ok = await checkHealth(base);
      if (ok) {
        try {
          const { voices: v, defaultId } = await fetchVoices(base);
          setVoices(v);
          setVoiceId(defaultId);
          setBootNote(null);
        } catch {
          setBootNote("API reachable but voices failed to load.");
        }
      } else {
        setBootNote(
          "Set your PC’s LAN IP in Settings, then run python run_web.py"
        );
        setPage("settings");
      }
      setBooting(false);
    })();
  }, []);

  const selectedVoice = voices.find((v) => v.id === voiceId) || voices[0];

  if (booting) {
    return (
      <SafeAreaProvider>
        <SafeAreaView style={styles.boot}>
          <ActivityIndicator size="large" color="#7c3aed" />
          <Text style={styles.bootText}>Starting Sign to Speech…</Text>
          <StatusBar style="dark" />
        </SafeAreaView>
      </SafeAreaProvider>
    );
  }

  return (
    <SafeAreaProvider>
      <SafeAreaView style={styles.root}>
        <StatusBar style="dark" />
        {page === "live" ? (
          <LiveScreen
            apiBase={apiBase}
            voice={selectedVoice}
            recognitionMode={recognitionMode}
            onModeChange={(m) => {
              setRecognitionMode(m);
              AsyncStorage.setItem(MODE_KEY, m).catch(() => {});
            }}
            onOpenSettings={() => setPage("settings")}
          />
        ) : (
          <SettingsScreen
            apiBase={apiBase}
            voiceId={voiceId}
            voices={voices}
            recognitionMode={recognitionMode}
            onModeChange={(m) => {
              setRecognitionMode(m);
              AsyncStorage.setItem(MODE_KEY, m).catch(() => {});
            }}
            onBack={() => setPage("live")}
            onSelectVoice={setVoiceId}
            onSaved={(base, v, defaultId) => {
              setApiBase(base);
              setVoices(v);
              setVoiceId((prev) =>
                v.some((x) => x.id === prev) ? prev : defaultId
              );
              setBootNote(null);
            }}
          />
        )}
        {bootNote && page === "live" && (
          <View style={styles.banner}>
            <Text style={styles.bannerText}>{bootNote}</Text>
          </View>
        )}
      </SafeAreaView>
    </SafeAreaProvider>
  );
}

const styles = StyleSheet.create({
  root: {
    flex: 1,
    backgroundColor: "#f5f7fb",
  },
  boot: {
    flex: 1,
    alignItems: "center",
    justifyContent: "center",
    backgroundColor: "#f5f7fb",
    gap: 12,
  },
  bootText: {
    color: "#6b7280",
    fontSize: 14,
  },
  banner: {
    position: "absolute",
    left: 16,
    right: 16,
    bottom: 24,
    backgroundColor: "#fff7ed",
    borderColor: "#fed7aa",
    borderWidth: 1,
    borderRadius: 12,
    padding: 12,
  },
  bannerText: {
    color: "#9a3412",
    fontSize: 12,
    lineHeight: 17,
  },
});
