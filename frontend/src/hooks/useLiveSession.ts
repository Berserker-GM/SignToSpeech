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

export function useLiveSession(enabled: boolean, recognitionMode: string = "STATIC") {
  const wsRef = useRef<WebSocket | null>(null);
  const modeRef = useRef(recognitionMode);
  const [connected, setConnected] = useState(false);
  const [frame, setFrame] = useState<FrameResult>({ hand_detected: false });
  const [sentence, setSentence] = useState<string[]>([]);
  const [mode, setModeState] = useState(recognitionMode);
  const [serverVocab, setServerVocab] = useState<string[]>([]);

  modeRef.current = recognitionMode;

  useEffect(() => {
    if (!enabled) {
      setConnected(false);
      return;
    }

    const ws = new WebSocket(wsUrl());
    wsRef.current = ws;

    ws.onopen = () => {
      setConnected(true);
      // Force STATIC/AUTO/DYNAMIC immediately — engine may still be on an old mode
      ws.send(JSON.stringify({ type: "set_mode", mode: modeRef.current }));
    };
    ws.onclose = () => {
      setConnected(false);
      wsRef.current = null;
    };
    ws.onerror = () => setConnected(false);
    ws.onmessage = (ev) => {
      const data = JSON.parse(ev.data);
      if (data.type === "hello") {
        if (Array.isArray(data.static_signs)) setServerVocab(data.static_signs);
        if (data.mode) setModeState(data.mode);
        // Re-assert client mode after hello (server default may differ)
        ws.send(JSON.stringify({ type: "set_mode", mode: modeRef.current }));
        return;
      }
      if (data.type === "frame") {
        setFrame(data);
        if (Array.isArray(data.sentence)) setSentence(data.sentence);
        if (data.mode) setModeState(data.mode);
      }
      if (data.type === "cleared") setSentence(data.sentence ?? []);
      if (data.type === "mode" && data.mode) setModeState(data.mode);
    };

    return () => {
      ws.close();
      wsRef.current = null;
    };
  }, [enabled]);

  // Push mode changes while connected
  useEffect(() => {
    const ws = wsRef.current;
    if (!ws || ws.readyState !== WebSocket.OPEN) return;
    ws.send(JSON.stringify({ type: "set_mode", mode: recognitionMode }));
    setModeState(recognitionMode);
  }, [recognitionMode]);

  const sendLandmarks = useCallback((landmarks: number[] | null) => {
    const ws = wsRef.current;
    if (!ws || ws.readyState !== WebSocket.OPEN) return;

    // Pin mode every frame so web can't silently stay on AUTO/DYNAMIC
    const payload: { type: string; landmarks: number[] | null; mode: string } = {
      type: "frame",
      landmarks,
      mode: modeRef.current,
    };
    ws.send(JSON.stringify(payload));
  }, []);

  const setMode = useCallback((next: string) => {
    wsRef.current?.send(JSON.stringify({ type: "set_mode", mode: next }));
    setModeState(next);
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
    mode,
    serverVocab,
    sendLandmarks,
    setMode,
    clearSentence,
    resetSession,
  };
}
