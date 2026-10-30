"""Handing a call to a person when the voice channel itself fails (STT or TTS down twice).

Uses the same system handoff the agent uses for its own failures (`gate.escalate_now`):
through the policy engine, the ledger and the audit log, with no change to the agent.
"""

import asyncio

from app.agent.loop import Agent
from app.agent.session import SessionBusyError
from app.domain.enums import EscalationReason
from app.policy import gate
from app.tools.context import ToolContext


async def escalate_voice_failure(agent: Agent, session_id: str, what_happened: str) -> str | None:
    """The ticket reference, or None if even the handoff failed. Never raises."""
    for _ in range(20):  # the agent's turn may still be finishing (it's been cancelled)
        try:
            async with agent.store.lock(session_id):
                session = await agent.store.load(session_id)
                if session is None:
                    return None
                if session.escalation_reference:
                    return session.escalation_reference
                async with agent.db_factory() as db:
                    ctx = ToolContext(
                        db=db,
                        state=session.state,
                        conversation_id=session.conversation_id,
                        clock=agent.clock,
                    )
                    result = await gate.escalate_now(
                        ctx,
                        reason=EscalationReason.TOOL_FAILURE,
                        issue=f"Voice channel failure: {what_happened}"[:2000],
                        order_number=session.current_order_number,
                        note=f"system: voice: {what_happened}",
                    )
                if not (result.ok and result.tool_result is not None):
                    return None
                session.escalation_reference = result.tool_result.data["ticket_reference"]
                await agent.store.save(session)
                return session.escalation_reference
        except SessionBusyError:
            await asyncio.sleep(0.1)
        except Exception:
            return None
    return None
