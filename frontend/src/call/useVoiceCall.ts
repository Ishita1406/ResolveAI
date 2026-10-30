// A voice call over WebRTC: the microphone goes to the backend, the agent's voice comes back
// as a remote audio track, and the backend's events arrive on a data channel. Signalling is
// one request (non-trickle ICE), as backend/app/voice/webrtc.py expects.

import { useCallback, useEffect, useReducer, useRef, useState } from "react";
import { ApiError, api } from "../api/client";
import type { VoiceEvent } from "../api/types";
import { callReducer, initialCallState } from "./callState";

export type MicPermission = "unknown" | "prompt" | "granted" | "denied" | "unsupported";

export function micErrorMessage(error: unknown): string {
  const name = error instanceof DOMException ? error.name : "";
  switch (name) {
    case "NotAllowedError":
    case "SecurityError":
      return "Microphone access is blocked. Allow it in your browser's site settings, then try again.";
    case "NotFoundError":
      return "No microphone was found. Connect one and try again.";
    case "NotReadableError":
      return "Your microphone is busy in another app. Close it there and try again.";
    default:
      return error instanceof Error ? error.message : "The microphone couldn't be opened.";
  }
}

function waitForIceGathering(pc: RTCPeerConnection, timeoutMs = 3000): Promise<void> {
  return new Promise((resolve) => {
    if (pc.iceGatheringState === "complete") return resolve();
    const done = () => {
      if (pc.iceGatheringState === "complete") {
        pc.removeEventListener("icegatheringstatechange", done);
        resolve();
      }
    };
    pc.addEventListener("icegatheringstatechange", done);
    setTimeout(resolve, timeoutMs); // send what we have rather than hang
  });
}

export function useVoiceCall() {
  const [state, dispatch] = useReducer(callReducer, initialCallState);
  const [permission, setPermission] = useState<MicPermission>("unknown");
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const pcRef = useRef<RTCPeerConnection | null>(null);
  const streamRef = useRef<MediaStream | null>(null);

  // Show the microphone permission before the call (not every browser can say).
  useEffect(() => {
    if (!navigator.mediaDevices?.getUserMedia) {
      setPermission("unsupported");
      return;
    }
    let status: PermissionStatus | null = null;
    const update = () => status && setPermission(status.state as MicPermission);
    navigator.permissions
      ?.query({ name: "microphone" as PermissionName })
      .then((s) => {
        status = s;
        update();
        s.addEventListener("change", update);
      })
      .catch(() => setPermission("unknown"));
    return () => status?.removeEventListener("change", update);
  }, []);

  const release = useCallback(() => {
    pcRef.current?.close();
    streamRef.current?.getTracks().forEach((t) => t.stop());
    pcRef.current = null;
    streamRef.current = null;
    if (audioRef.current) audioRef.current.srcObject = null;
  }, []);

  const end = useCallback(() => {
    release();
    dispatch({ kind: "ended" });
  }, [release]);

  const start = useCallback(async () => {
    if (!navigator.mediaDevices?.getUserMedia) {
      dispatch({
        kind: "failed",
        message: "This page needs a secure origin (https or localhost) to use the microphone.",
      });
      return;
    }
    dispatch({ kind: "connecting" });
    let stream: MediaStream;
    try {
      stream = await navigator.mediaDevices.getUserMedia({
        audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true },
      });
      setPermission("granted");
    } catch (error) {
      if (error instanceof DOMException && error.name === "NotAllowedError") setPermission("denied");
      dispatch({ kind: "failed", message: micErrorMessage(error) });
      return;
    }
    streamRef.current = stream;
    const pc = new RTCPeerConnection();
    pcRef.current = pc;
    stream.getTracks().forEach((track) => pc.addTrack(track, stream));
    pc.ontrack = (event) => {
      if (audioRef.current) {
        audioRef.current.srcObject = event.streams[0] ?? new MediaStream([event.track]);
        void audioRef.current.play().catch(() => undefined); // autoplay after a click is allowed
      }
    };
    pc.onconnectionstatechange = () => {
      if (pc.connectionState === "failed") {
        release();
        dispatch({ kind: "failed", message: "The call's connection was lost." });
      }
    };
    const events = pc.createDataChannel("events");
    events.onmessage = (message) => {
      try {
        dispatch({ kind: "event", event: JSON.parse(String(message.data)) as VoiceEvent });
      } catch {
        // a malformed frame: ignore it rather than end the call
      }
    };
    try {
      await pc.setLocalDescription(await pc.createOffer());
      await waitForIceGathering(pc);
      const local = pc.localDescription;
      if (!local) throw new Error("No local description");
      const answer = await api.voiceOffer({ sdp: local.sdp, type: local.type });
      await pc.setRemoteDescription({ sdp: answer.sdp, type: answer.type });
      dispatch({ kind: "connected", sessionId: answer.session_id });
    } catch (error) {
      release();
      const message =
        error instanceof ApiError ? error.message : "The call couldn't be connected.";
      dispatch({ kind: "failed", message });
    }
  }, [release]);

  useEffect(() => release, [release]); // hang up when the page goes away

  return { state, permission, audioRef, start, end };
}
