"""Paso 1: cadena LCEL sobre las políticas. Sin memoria ni herramientas.

Ejecutar: python paso1.py
"""

from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable

from common import (
    POLICIES_CONTEXT,
    build_model,
    build_parser,
    load_policies,
    require_question,
    run_chat,
)

SYSTEM_TEMPLATE = """\
Eres el asistente de atención al cliente de EventHub.
Respondes exclusivamente con la información contenida en las políticas de abajo.
Reglas:
- No uses conocimiento externo ni completes datos que no aparezcan en el texto.
- No inventes plazos, importes, excepciones ni estados de pedido.
- Si la pregunta no puede responderse con las políticas, di claramente que no dispones de esa información.
- Responde en el mismo idioma que la pregunta, de forma breve y directa.
""" + POLICIES_CONTEXT

_cached_chain: Runnable | None = None


def build_prompt() -> ChatPromptTemplate:
    """Sistema con las políticas y la pregunta. No hay historial."""
    return ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_TEMPLATE),
            ("human", "{question}"),
        ]
    )


def build_chain(policies: str | None = None) -> Runnable:
    """Compone prompt | model | parser."""
    policies_text = load_policies() if policies is None else policies
    prompt = build_prompt().partial(policies=policies_text)
    return prompt | build_model() | build_parser()


def answer_question(question: str) -> str:
    """Ejecuta la cadena del paso 1. Cada pregunta es independiente."""
    global _cached_chain
    text = require_question(question)
    if _cached_chain is None:
        _cached_chain = build_chain()
    return _cached_chain.invoke({"question": text})


def main() -> None:
    run_chat(
        answer_question,
        use_session=False,
        title="Paso 1. Políticas de EventHub, sin memoria ni herramientas.",
    )


if __name__ == "__main__":
    main()
