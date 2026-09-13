export type HistoryEntry = {
  id: string;
  raw: string;
  corrected: string;
  words: string[];
  savedAt: number;
};

const KEY = "sts_conversation_history";
const DEMO_FLAG = "sts_history_demo_seeded_v2";
const MAX_ENTRIES = 200;

/** Timestamps for this calendar week, excluding today (newest → oldest). */
function demoTimestampsThisWeek(count: number): number[] {
  const startOfToday = new Date();
  startOfToday.setHours(0, 0, 0, 0);

  // Monday 00:00 of the current week (Sun=0 → treat week as Mon–Sun)
  const day = startOfToday.getDay();
  const daysFromMonday = day === 0 ? 6 : day - 1;
  const weekStart = new Date(startOfToday);
  weekStart.setDate(weekStart.getDate() - daysFromMonday);

  // Candidate days: Mon … yesterday (never today)
  const dayStarts: Date[] = [];
  for (let d = new Date(weekStart); d < startOfToday; d.setDate(d.getDate() + 1)) {
    dayStarts.push(new Date(d));
  }

  // If today is Monday (no earlier day this week), fall back to last week Mon–Sun
  if (dayStarts.length === 0) {
    for (let i = 7; i >= 1; i--) {
      const d = new Date(startOfToday);
      d.setDate(d.getDate() - i);
      dayStarts.push(d);
    }
  }

  const times = [10, 11, 14, 15, 16, 18]; // varied hours
  const out: number[] = [];
  for (let i = 0; i < count; i++) {
    // Prefer recent earlier days first
    const dayIdx = Math.min(i, dayStarts.length - 1);
    const dayDate = dayStarts[dayStarts.length - 1 - dayIdx];
    const t = new Date(dayDate);
    t.setHours(times[i % times.length], 15 + i * 7, 0, 0);
    out.push(t.getTime());
  }
  return out;
}

/** Sample raw gloss → DeepSeek-style sentences (seeded once on first visit). */
const DEMO_ENTRIES: Omit<HistoryEntry, "id" | "savedAt">[] = [
  {
    raw: "HELP PLEASE WATER",
    corrected: "I need water, please.",
    words: ["HELP", "PLEASE", "WATER"],
  },
  {
    raw: "HELLO HELP",
    corrected: "Hello, I need help.",
    words: ["HELLO", "HELP"],
  },
  {
    raw: "WHERE RESTROOM",
    corrected: "Where is the restroom?",
    words: ["WHERE", "RESTROOM"],
  },
  {
    raw: "I WANT EAT FOOD",
    corrected: "I would like something to eat.",
    words: ["I", "WANT", "EAT", "FOOD"],
  },
  {
    raw: "THANK YOU FRIEND",
    corrected: "Thank you, my friend.",
    words: ["THANK", "YOU", "FRIEND"],
  },
  {
    raw: "FEEL BAD NEED DOCTOR",
    corrected: "I feel unwell and need a doctor.",
    words: ["FEEL", "BAD", "NEED", "DOCTOR"],
  },
];

function seedDemoHistory(): HistoryEntry[] {
  const stamps = demoTimestampsThisWeek(DEMO_ENTRIES.length);
  const seeded: HistoryEntry[] = DEMO_ENTRIES.map((entry, i) => ({
    ...entry,
    id: `demo-${i + 1}`,
    savedAt: stamps[i],
  }));
  localStorage.setItem(KEY, JSON.stringify(seeded));
  localStorage.setItem(DEMO_FLAG, "1");
  return seeded;
}

function refreshDemoDates(entries: HistoryEntry[]): HistoryEntry[] {
  const demos = entries.filter((e) => e.id.startsWith("demo-"));
  if (!demos.length) return entries;
  const stamps = demoTimestampsThisWeek(demos.length);
  const stampById = new Map(
    demos.map((e, i) => [e.id, stamps[i]] as const)
  );
  return entries.map((e) =>
    stampById.has(e.id) ? { ...e, savedAt: stampById.get(e.id)! } : e
  );
}

export function loadHistory(): HistoryEntry[] {
  try {
    const raw = localStorage.getItem(KEY);
    // First visit (or empty before demos existed) — seed showcase examples once
    if (raw === null) return seedDemoHistory();
    const parsed = JSON.parse(raw) as HistoryEntry[];
    if (!Array.isArray(parsed)) return [];
    if (parsed.length === 0 && localStorage.getItem(DEMO_FLAG) !== "1") {
      return seedDemoHistory();
    }
    // One-time bump: move existing demo timestamps off today onto this week
    if (localStorage.getItem(DEMO_FLAG) !== "1" && parsed.some((e) => e.id.startsWith("demo-"))) {
      const next = refreshDemoDates(parsed);
      localStorage.setItem(KEY, JSON.stringify(next));
      localStorage.setItem(DEMO_FLAG, "1");
      return next;
    }
    return parsed;
  } catch {
    return [];
  }
}

export function saveHistoryEntry(entry: Omit<HistoryEntry, "id" | "savedAt">): HistoryEntry {
  const full: HistoryEntry = {
    ...entry,
    id: `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
    savedAt: Date.now(),
  };
  const list = loadHistory();
  // Avoid exact duplicate of the latest save
  if (
    list[0] &&
    list[0].corrected === full.corrected &&
    list[0].raw === full.raw &&
    Date.now() - list[0].savedAt < 3000
  ) {
    return list[0];
  }
  const next = [full, ...list].slice(0, MAX_ENTRIES);
  localStorage.setItem(KEY, JSON.stringify(next));
  return full;
}

export function deleteHistoryEntry(id: string): HistoryEntry[] {
  const next = loadHistory().filter((e) => e.id !== id);
  localStorage.setItem(KEY, JSON.stringify(next));
  return next;
}

export function clearHistory(): void {
  localStorage.setItem(KEY, "[]");
  localStorage.setItem(DEMO_FLAG, "1");
}

export function formatHistoryTime(ts: number): string {
  try {
    return new Date(ts).toLocaleString(undefined, {
      dateStyle: "medium",
      timeStyle: "short",
    });
  } catch {
    return "";
  }
}
