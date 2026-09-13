import { useCallback, useEffect, useRef, useState } from "react";
import { toWsUrl } from "../lib/config";

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

export function useLiveSession(
  apiBase: string | null,
  enabled: boolean,
  recognitionMode: string = "AUTO"
) {
  const wsRef = useRef<WebSocket | null>(null);
  const modeRef = useRef(recognitionMode);
  const [connected, setConnected] = useState(false);
  const [frame, setFrame] = useState<FrameResult>({ hand_detected: false });
  const [sentence, setSentence] = useState<string[]>([]);

  modeRef.current = recognitionMode;

  useEffect(() => {
    if (!enabled || !apiBase) {
      setConnected(false);
      return;
    }

    let closed = false;
    let ws: WebSocket;
    try {
      ws = new WebSocket(toWsUrl(apiBase));
    } catch {
      setConnected(false);
      return;
    }
    wsRef.current = ws;

    ws.onopen = () => {
      if (!closed) {
        setConnected(true);
        ws.send(JSON.stringify({ type: "set_mode", mode: modeRef.current }));
      }
    };
    ws.onclose = () => {
      if (!closed) setConnected(false);
    };
    ws.onerror = () => {
      if (!closed) setConnected(false);
    };
    ws.onmessage = (ev) => {
      try {
        const data = JSON.parse(String(ev.data));
        if (data.type === "frame") {
          setFrame(data);
          if (data.sentence) setSentence(data.sentence);
        }
        if (data.type === "cleared") setSentence(data.sentence ?? []);
        if (data.type === "mode" && data.mode) {
          setFrame((f) => ({ ...f, mode: data.mode }));
        }
      } catch {
        /* ignore malformed */
      }
    };

    return () => {
      closed = true;
      ws.close();
      wsRef.current = null;
      setConnected(false);
    };
  }, [apiBase, enabled]);

  useEffect(() => {
    const ws = wsRef.current;
    if (!ws || ws.readyState !== WebSocket.OPEN) return;
    ws.send(JSON.stringify({ type: "set_mode", mode: recognitionMode }));
  }, [recognitionMode]);

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
