import { Platform } from "react-native";
import AsyncStorage from "@react-native-async-storage/async-storage";
import Constants from "expo-constants";

const KEY = "sts_api_base_url_v2";
const LEGACY_BAD_DEFAULT = "http://192.168.1.100:8000";

/** Best-effort API base: web → localhost, phone → same host as Expo Metro. */
export function getDefaultApiBase(): string {
  if (Platform.OS === "web") {
    if (typeof window !== "undefined" && window.location?.hostname) {
      const host = window.location.hostname;
      if (host === "localhost" || host === "127.0.0.1") {
        return "http://127.0.0.1:8000";
      }
      return `http://${host}:8000`;
    }
    return "http://127.0.0.1:8000";
  }

  const hostUri =
    Constants.expoConfig?.hostUri ||
    (Constants as { manifest2?: { extra?: { expoGo?: { debuggerHost?: string } } } })
      .manifest2?.extra?.expoGo?.debuggerHost ||
    (Constants as { manifest?: { debuggerHost?: string } }).manifest?.debuggerHost;

  if (hostUri) {
    const host = String(hostUri).split(":")[0];
    if (host && host !== "localhost" && host !== "127.0.0.1") {
      return `http://${host}:8000`;
    }
  }

  return "http://127.0.0.1:8000";
}

/** @deprecated use getDefaultApiBase() — kept for Settings placeholder text */
export const DEFAULT_API_BASE = getDefaultApiBase();

export async function getApiBaseUrl(): Promise<string> {
  const saved = await AsyncStorage.getItem(KEY);
  if (saved && saved !== LEGACY_BAD_DEFAULT) {
    return saved.replace(/\/$/, "");
  }
  // Also clear legacy key if present
  await AsyncStorage.removeItem("sts_api_base_url");
  return getDefaultApiBase();
}

export async function setApiBaseUrl(url: string): Promise<string> {
  const cleaned = url.trim().replace(/\/$/, "");
  await AsyncStorage.setItem(KEY, cleaned);
  return cleaned;
}

export function toWsUrl(httpBase: string): string {
  const u = httpBase.replace(/\/$/, "");
  if (u.startsWith("https://")) return u.replace(/^https/, "wss") + "/ws/live";
  if (u.startsWith("http://")) return u.replace(/^http/, "ws") + "/ws/live";
  return `ws://${u}/ws/live`;
}
