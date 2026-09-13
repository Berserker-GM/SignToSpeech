import { createAudioPlayer, setAudioModeAsync } from "expo-audio";
import * as Speech from "expo-speech";
import {
  audioSrcFromResult,
  speakText,
  type SpeakResult,
  type Voice,
} from "./api";

let activePlayer: ReturnType<typeof createAudioPlayer> | null = null;

export async function speakWithExpo(text: string, volume = 0.9): Promise<void> {
  return new Promise((resolve, reject) => {
    Speech.stop();
    Speech.speak(text, {
      language: "en-US",
      volume,
      rate: 0.95,
      onDone: () => resolve(),
      onStopped: () => resolve(),
      onError: () => reject(new Error("Device speech failed")),
    });
  });
}

export function stopSpeech(): void {
  Speech.stop();
  try {
    activePlayer?.pause();
    activePlayer?.release();
  } catch {
    /* ignore */
  }
  activePlayer = null;
}

export async function playSpeakResult(
  base: string,
  text: string,
  voice: Voice | undefined,
  volume: number
): Promise<{ warning?: string | null }> {
  const useDevice =
    !voice || voice.id === "browser" || voice.provider === "browser";

  if (useDevice) {
    await speakWithExpo(text, volume);
    return {};
  }

  let result: SpeakResult;
  try {
    result = await speakText(base, text, voice.id);
  } catch {
    await speakWithExpo(text, volume);
    return { warning: "ElevenLabs unavailable — used device voice." };
  }

  if (result.provider === "browser") {
    await speakWithExpo(result.text || text, volume);
    return { warning: result.warning };
  }

  const src = audioSrcFromResult(result);
  if (!src) {
    await speakWithExpo(text, volume);
    return { warning: "No audio returned — used device voice." };
  }

  await setAudioModeAsync({
    playsInSilentMode: true,
    interruptionMode: "duckOthers",
  });

  stopSpeech();
  const player = createAudioPlayer({ uri: src });
  activePlayer = player;
  player.volume = Math.min(1, Math.max(0, volume));
  player.play();

  await new Promise<void>((resolve) => {
    const sub = player.addListener("playbackStatusUpdate", (status) => {
      if (status.didJustFinish) {
        sub.remove();
        try {
          player.release();
        } catch {
          /* ignore */
        }
        if (activePlayer === player) activePlayer = null;
        resolve();
      }
    });
  });

  return { warning: result.warning };
}
