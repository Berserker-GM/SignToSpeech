const API = import.meta.env.DEV ? "" : "";

export type Voice = {
  id: string;
  name: string;
  gender: string;
  provider?: "browser" | "elevenlabs";
};

export type SpeakResult = {
  provider: "browser" | "elevenlabs";
  text?: string;
  audio_base64?: string;
  mime?: string;
  warning?: string | null;
};

export async function fetchVoices(): Promise<{ voices: Voice[]; defaultId: string }> {
  const res = await fetch(`${API}/api/voices`);
  const data = await res.json();
  return {
    voices: data.voices,
    defaultId: data.default_voice_id || "browser",
  };
}

export async function correctSentence(words: string[]) {
  const res = await fetch(`${API}/api/correct`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ words }),
  });
  if (!res.ok) throw new Error("Grammar correction failed");
  return res.json() as Promise<{
    raw: string;
    corrected: string;
    grammar_active: boolean;
  }>;
}

export async function speakText(text: string, voiceId: string): Promise<SpeakResult> {
  const res = await fetch(`${API}/api/speak`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text, voice_id: voiceId }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    const detail = (err as { detail?: string | { message?: string } }).detail;
    const msg =
      typeof detail === "string"
        ? detail
        : (detail as { message?: string })?.message || "TTS failed";
    throw new Error(msg);
  }
  return res.json() as Promise<SpeakResult>;
}

export function audioSrcFromResult(result: SpeakResult): string | null {
  if (result.provider === "elevenlabs" && result.audio_base64) {
    return `data:${result.mime || "audio/mpeg"};base64,${result.audio_base64}`;
  }
  return null;
}

export async function clearSession(): Promise<void> {
  await fetch(`${API}/api/session/reset`, { method: "POST" });
}

export function wsUrl(): string {
  const proto = window.location.protocol === "https:" ? "wss" : "ws";
  if (import.meta.env.DEV) return `${proto}://${window.location.hostname}:8000/ws/live`;
  return `${proto}://${window.location.host}/ws/live`;
}
