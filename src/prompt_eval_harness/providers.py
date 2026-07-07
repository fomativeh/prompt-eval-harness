from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod

from openai import OpenAI

from prompt_eval_harness.models import EvalCase


class BaseProvider(ABC):
    @abstractmethod
    def generate(self, prompt_text: str, case: EvalCase, model: str) -> str:
        raise NotImplementedError


class MockProvider(BaseProvider):
    def generate(self, prompt_text: str, case: EvalCase, model: str) -> str:
        payload = {
            "category": case.expected_category,
            "urgency": case.expected_urgency,
            "should_refuse": case.expected_should_refuse,
            "reply_tone": case.expected_reply_tone,
            "response": self._build_response(case),
        }
        if "support assistant." in prompt_text and case.id in {"case_009", "case_016", "case_021", "case_024"}:
            payload["should_refuse"] = False
            payload["reply_tone"] = "neutral"
        return json.dumps(payload)

    @staticmethod
    def _build_response(case: EvalCase) -> str:
        if case.expected_should_refuse:
            return "I can't help with that request, but I can point you to safe account and security support options."
        if case.expected_reply_tone == "empathetic":
            return "I'm sorry you're running into this. I'll help route this to the right team and suggest the next step."
        return "Thanks for the details. I'll route this to the right team and share the next step."


class OpenAIProvider(BaseProvider):
    def __init__(self) -> None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY is required to use the openai provider.")
        self.client = OpenAI(api_key=api_key)

    def generate(self, prompt_text: str, case: EvalCase, model: str) -> str:
        response = self.client.chat.completions.create(
            model=model,
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": prompt_text},
                {"role": "user", "content": case.customer_message},
            ],
        )
        return response.choices[0].message.content or ""


def get_provider(name: str) -> BaseProvider:
    if name == "mock":
        return MockProvider()
    if name == "openai":
        return OpenAIProvider()
    raise ValueError(f"Unsupported provider: {name}")
