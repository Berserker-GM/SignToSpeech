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

export type CorrectResult = {
  raw: string;
  corrected: string;
  grammar_active: boolean;
  error?: string;
};

export async function fetchParams(base: string): Promise<Record<string, unknown>> {
  const res = await fetch(`${base}/api/params`);
  if (!res.ok) throw new Error("Failed to load params");
  return res.json();
}

async function jsonFetch<T>(url: string, init?: RequestInit): Promise<T> {
  const res = await fetch(url, init);
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    const detail = (err as { detail?: string }).detail;
    throw new Error(detail || `Request failed (${res.status})`);
  }
  return res.json() as Promise<T>;
}

export async function checkHealth(base: string): Promise<boolean> {
  try {
    const data = await jsonFetch<{ ok: boolean }>(`${base}/api/health`);
    return !!data.ok;
  } catch {
    return false;
  }
}

export async function fetchVoices(base: string): Promise<{
  voices: Voice[];
  defaultId: string;
}> {
  const data = await jsonFetch<{
    voices: Voice[];
    default_voice_id?: string;
  }>(`${base}/api/voices`);
  return {
    voices: data.voices,
    defaultId: data.default_voice_id || "browser",
  };
}

export async function correctSentence(
  base: string,
  words: string[]
): Promise<CorrectResult> {
  return jsonFetch(`${base}/api/correct`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ words }),
  });
}

export async function speakText(
  base: string,
  text: string,
  voiceId: string
): Promise<SpeakResult> {
  return jsonFetch(`${base}/api/speak`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text, voice_id: voiceId }),
  });
}

export async function clearSession(base: string): Promise<void> {
  await fetch(`${base}/api/session/reset`, { method: "POST" });
}

export function audioSrcFromResult(result: SpeakResult): string | null {
  if (result.provider === "elevenlabs" && result.audio_base64) {
    return `data:${result.mime || "audio/mpeg"};base64,${result.audio_base64}`;
  }
  return null;
}
