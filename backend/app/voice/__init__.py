"""Real-time voice around the agent: WebRTC transport, VAD and turn-taking, streaming STT
(Cartesia Ink) and TTS (Cartesia Sonic), barge-in, fillers, and failure handling.

The agent is used exactly as the text channel uses it (`agent.respond()`); nothing here
changes its logic, policy or safety checks.
"""
