import { useEffect, useRef } from "react";
import { Platform, StyleSheet, View } from "react-native";
import { WebView, type WebViewMessageEvent } from "react-native-webview";
import { HAND_TRACKER_HTML } from "../tracker/handTrackerHtml";

type Props = {
  paused: boolean;
  onLandmarks: (landmarks: number[] | null, handDetected: boolean) => void;
  onReady?: () => void;
  onCameraStatus?: (status: "on" | "off" | "denied", message?: string) => void;
  onError?: (message: string) => void;
};

export function HandTrackerWebView({
  paused,
  onLandmarks,
  onReady,
  onCameraStatus,
  onError,
}: Props) {
  const webRef = useRef<WebView>(null);

  useEffect(() => {
    const payload = JSON.stringify({ type: paused ? "pause" : "resume" });
    webRef.current?.injectJavaScript(
      `window.postMessage(${JSON.stringify(payload)}, "*"); true;`
    );
  }, [paused]);

  const onMessage = (event: WebViewMessageEvent) => {
    try {
      const data = JSON.parse(event.nativeEvent.data);
      if (data.type === "landmarks") {
        onLandmarks(data.landmarks ?? null, !!data.handDetected);
      } else if (data.type === "ready") {
        onReady?.();
      } else if (data.type === "camera") {
        onCameraStatus?.(data.status, data.message);
      } else if (data.type === "error") {
        onError?.(data.message || "Tracker error");
      }
    } catch {
      /* ignore */
    }
  };

  return (
    <View style={styles.wrap}>
      <WebView
        ref={webRef}
        originWhitelist={["*"]}
        source={{ html: HAND_TRACKER_HTML, baseUrl: "https://cdn.jsdelivr.net" }}
        style={styles.webview}
        onMessage={onMessage}
        allowsInlineMediaPlayback
        mediaPlaybackRequiresUserAction={false}
        mediaCapturePermissionGrantType="grant"
        javaScriptEnabled
        domStorageEnabled
        allowsFullscreenVideo
        mixedContentMode="always"
        {...(Platform.OS === "android"
          ? { androidLayerType: "hardware" as const }
          : {})}
      />
    </View>
  );
}

const styles = StyleSheet.create({
  wrap: {
    flex: 1,
    backgroundColor: "#0f1115",
    overflow: "hidden",
    borderRadius: 16,
  },
  webview: {
    flex: 1,
    backgroundColor: "#0f1115",
  },
});
