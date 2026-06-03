let activeUtterance: SpeechSynthesisUtterance | null = null;

export function browserVoices(): SpeechSynthesisVoice[] {
  return speechSynthesis.getVoices().filter((v) => v.lang.startsWith("en"));
}

export function speakBrowser(
  text: string,
  volume: number,
  lang = "en-US"
): Promise<void> {
  return new Promise((resolve, reject) => {
    if (!("speechSynthesis" in window)) {
      reject(new Error("Browser speech not supported"));
      return;
    }

    speechSynthesis.cancel();
    const utterance = new SpeechSynthesisUtterance(text);
    utterance.lang = lang;
    utterance.volume = Math.min(1, Math.max(0, volume));
    utterance.rate = 1;

    const voices = browserVoices();
    const preferred =
      voices.find((v) => v.name.includes("Google") && v.lang.startsWith("en")) ||
      voices.find((v) => v.lang.startsWith("en")) ||
      voices[0];
    if (preferred) utterance.voice = preferred;

    utterance.onend = () => {
      activeUtterance = null;
      resolve();
    };
    utterance.onerror = () => {
      activeUtterance = null;
      reject(new Error("Browser speech failed"));
    };

    activeUtterance = utterance;
    speechSynthesis.speak(utterance);
  });
}

export function stopBrowserSpeech(): void {
  speechSynthesis.cancel();
  activeUtterance = null;
}
