import { useCallback, useEffect, useRef, useState } from "react";
import { wsUrl } from "../lib/api";

export type FrameResult = {
  hand_detected: boolean;
  static_pred?: string;
  static_conf?: number;
  dynamic_pred?: string | null;
  dynamic_conf?: number;
  final_pred?: string | null;
  final_conf?: number;
  final_source?: string;
  hold_progress?: number;
  confirmed?: string | null;
  sentence?: string[];
  sequence_len?: number;
  sequence_max?: number;
  mode?: string;
};

export function useLiveSession(enabled: boolean) {
  const wsRef = useRef<WebSocket | null>(null);
  const [connected, setConnected] = useState(false);
  const [frame, setFrame] = useState<FrameResult>({ hand_detected: false });
  const [sentence, setSentence] = useState<string[]>([]);

  useEffect(() => {
    if (!enabled) return;

    const ws = new WebSocket(wsUrl());
    wsRef.current = ws;

    ws.onopen = () => setConnected(true);
    ws.onclose = () => setConnected(false);
    ws.onmessage = (ev) => {
      const data = JSON.parse(ev.data);
      if (data.type === "frame") {
        setFrame(data);
        if (data.sentence) setSentence(data.sentence);
      }
      if (data.type === "cleared") setSentence(data.sentence ?? []);
    };

    return () => {
      ws.close();
      wsRef.current = null;
    };
  }, [enabled]);

  const sendLandmarks = useCallback((landmarks: number[] | null) => {
    const ws = wsRef.current;
    if (!ws || ws.readyState !== WebSocket.OPEN) return;

    if (!landmarks) {
      ws.send(JSON.stringify({ type: "frame", landmarks: null }));
      return;
    }
    ws.send(JSON.stringify({ type: "frame", landmarks }));
  }, []);

  const setMode = useCallback((mode: string) => {
    wsRef.current?.send(JSON.stringify({ type: "set_mode", mode }));
  }, []);

  const clearSentence = useCallback(() => {
    wsRef.current?.send(JSON.stringify({ type: "clear" }));
    setSentence([]);
  }, []);

  const resetSession = useCallback(() => {
    wsRef.current?.send(JSON.stringify({ type: "reset" }));
    setSentence([]);
  }, []);

  return {
    connected,
    frame,
    sentence,
    sendLandmarks,
    setMode,
    clearSentence,
    resetSession,
  };
}
