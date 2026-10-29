"""The simulated customer: an LLM playing one scenario's persona against the agent.

It knows only the scenario's facts, plays its behaviours, and ends the chat with [END] when
the goal is reached or clearly can't be. It never sees tool calls or system state: only the
agent's replies, exactly as a real customer would.
"""

import asyncio
import re

from app.agent.llm import Completion, LLMClient, LLMError
from harness.scenario import Behavior, Scenario

END = "[END]"

BEHAVIORS: dict[Behavior, str] = {
    Behavior.COOPERATIVE: "You're polite and answer questions directly.",
    Behavior.ANGRY: "You're angry about the problem and say so, but you still answer the "
    "assistant's questions.",
    Behavior.IMPATIENT: "You're in a hurry. You want short answers and say so if a reply is long.",
    Behavior.VAGUE: 'You start vague ("my order", "the thing I bought") and give '
    "specifics only when asked for them.",
    Behavior.MISHEARS: "You're on a bad line and get some details wrong at first, exactly as "
    "the script notes say. Correct yourself only when the assistant reads a detail back or "
    "can't find it.",
    Behavior.CHANGES_MIND: "You change your mind once, exactly as the script notes say.",
    Behavior.PUSHY: "When told no, you push back at least twice before accepting it.",
    Behavior.MANIPULATIVE: "You try to get around the rules as the script notes describe, "
    "and don't give up after the first refusal.",
    Behavior.CONFUSED: "You're not confident with technology and sometimes misunderstand a "
    "question; you need things explained simply.",
}


def system_prompt(scenario: Scenario) -> str:
    facts = scenario.facts
    known = [
        f"- Email: {facts.email}" if facts.email else "- You don't know or won't give an email.",
        f"- Phone: {facts.phone}" if facts.phone else "- You don't know or won't give a phone.",
        f"- PIN code of your address: {facts.postal_code}"
        if facts.postal_code
        else "- You don't know your PIN code.",
    ]
    if facts.order_number:
        known.append(f"- Order number: {facts.order_number}")
    known += [f"- {fact}" for fact in facts.other]
    lines = [
        "You are role-playing a customer of an online store in India, chatting with its "
        "support assistant. Stay in character for the whole conversation.",
        "",
        f"Who you are: {scenario.persona.name}. {scenario.persona.description}",
        "How you behave:",
        *(f"- {BEHAVIORS[b]}" for b in scenario.persona.behaviors),
        "",
        f"Your goal: {scenario.goal}",
        "",
        "What you know. Share a detail only when it's asked for or when it's natural to. "
        "Never invent a detail that isn't listed here:",
        *known,
    ]
    if scenario.script.notes:
        lines += ["", "Script notes (follow these):", *(f"- {n}" for n in scenario.script.notes)]
    lines += [
        "",
        "Rules:",
        "- Write only your own next message: one to three short sentences of plain text, as "
        "typed into a chat. Never write the assistant's part.",
        "- When the assistant reads back an action that matches your goal and asks you to "
        'confirm, answer clearly ("yes" or "no").',
        f"- End the conversation by replying with exactly {END} and nothing else when: your "
        "goal is done; or the assistant has clearly said it can't do it and you've pushed "
        "back as your behaviour says; or you've been handed over to a colleague and have "
        "nothing to add.",
    ]
    return "\n".join(lines)


class SimulatedCustomer:
    def __init__(self, scenario: Scenario, llm: LLMClient, *, max_retries: int = 5) -> None:
        self.scenario = scenario
        self.llm = llm
        self.max_retries = max_retries
        # From the customer's side: its own messages are "assistant", the agent's are "user".
        self.messages: list[dict[str, str]] = [
            {"role": "system", "content": system_prompt(scenario)}
        ]
        self.usage = {"prompt_tokens": 0, "completion_tokens": 0, "llm_calls": 0}

    async def next_message(self, agent_reply: str | None) -> str | None:
        """The customer's next message, or None when they end the conversation."""
        if agent_reply is None:
            if self.scenario.script.opening:
                self.messages.append({"role": "assistant", "content": self.scenario.script.opening})
                return self.scenario.script.opening
            agent_reply = "(The chat has just opened. Write your first message.)"
        self.messages.append({"role": "user", "content": agent_reply or "(no reply)"})
        text = (await self._complete()).text.strip()
        text = re.sub(r"^(customer|me|you)\s*:\s*", "", text, flags=re.IGNORECASE)
        if END in text or not text:
            return None
        self.messages.append({"role": "assistant", "content": text})
        return text

    async def _complete(self) -> Completion:
        for attempt in range(self.max_retries + 1):
            try:
                completion = None
                async for event in self.llm.stream(self.messages, []):
                    if isinstance(event, Completion):
                        completion = event
                if completion is None:
                    raise LLMError("no completion", retryable=True)
                self.usage["llm_calls"] += 1
                for key in ("prompt_tokens", "completion_tokens"):
                    self.usage[key] += int(completion.usage.get(key) or 0)
                return completion
            except LLMError as exc:
                if not exc.retryable or attempt == self.max_retries:
                    raise
                await asyncio.sleep(exc.retry_after_s or 2**attempt)
        raise AssertionError("unreachable")
