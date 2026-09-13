import type { Voice } from "./api";

const SELECTED_KEY = "sts_selected_voice_id";

export type ShowcaseVoice = Voice & {
  style: string;
  accent: string;
  description: string;
  demo?: boolean;
};

/** Fake catalog for the Voices panel (preview uses browser TTS). */
export const DEMO_VOICES: ShowcaseVoice[] = [
  {
    id: "browser",
    name: "System Default",
    gender: "Neutral",
    provider: "browser",
    style: "Clear",
    accent: "Device",
    description: "Uses your browser’s built-in speech engine.",
    demo: false,
  },
  {
    id: "demo-aria",
    name: "Aria",
    gender: "Female",
    provider: "browser",
    style: "Soft",
    accent: "American",
    description: "Warm hospital-desk tone — calm and easy to follow.",
    demo: true,
  },
  {
    id: "demo-jordan",
    name: "Jordan",
    gender: "Male",
    provider: "browser",
    style: "Clear",
    accent: "American",
    description: "Bright public-space voice for announcements and help desks.",
    demo: true,
  },
  {
    id: "demo-priya",
    name: "Priya",
    gender: "Female",
    provider: "browser",
    style: "Warm",
    accent: "Indian English",
    description: "Friendly conversational style for everyday requests.",
    demo: true,
  },
  {
    id: "demo-marcus",
    name: "Marcus",
    gender: "Male",
    provider: "browser",
    style: "Professional",
    accent: "British",
    description: "Steady, formal delivery for clinics and transit desks.",
    demo: true,
  },
  {
    id: "demo-nova",
    name: "Nova",
    gender: "Neutral",
    provider: "browser",
    style: "Friendly",
    accent: "American",
    description: "Balanced modern voice that reads natural sentences well.",
    demo: true,
  },
  {
    id: "demo-sam",
    name: "Sam",
    gender: "Male",
    provider: "browser",
    style: "Calm",
    accent: "Australian",
    description: "Relaxed pace — good when someone needs a moment to listen.",
    demo: true,
  },
];

export function loadSelectedVoiceId(fallback = "browser"): string {
  try {
    return localStorage.getItem(SELECTED_KEY) || fallback;
  } catch {
    return fallback;
  }
}

export function saveSelectedVoiceId(id: string): void {
  try {
    localStorage.setItem(SELECTED_KEY, id);
  } catch {
    /* ignore */
  }
}

export function mergeVoiceLists(apiVoices: Voice[]): ShowcaseVoice[] {
  const demos = DEMO_VOICES.filter((d) => d.id !== "browser");
  const browser =
    DEMO_VOICES.find((d) => d.id === "browser") ||
    ({
      id: "browser",
      name: "System Default",
      gender: "Neutral",
      provider: "browser" as const,
      style: "Clear",
      accent: "Device",
      description: "Uses your browser’s built-in speech engine.",
    } satisfies ShowcaseVoice);

  const seen = new Set<string>([browser.id, ...demos.map((d) => d.id)]);
  const fromApi: ShowcaseVoice[] = apiVoices
    .filter((v) => !seen.has(v.id) && v.id !== "browser")
    .map((v) => ({
      ...v,
      style: v.provider === "elevenlabs" ? "Neural" : "System",
      accent: "—",
      description:
        v.provider === "elevenlabs"
          ? "Connected ElevenLabs voice from your API key."
          : "Available system voice.",
      demo: false,
    }));

  return [browser, ...fromApi, ...demos];
}
