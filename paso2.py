"""Paso 2: la cadena del paso 1 más historial por session_id. Sin herramientas.

Ejecutar: python paso2.py
"""

from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import Runnable

from common import (
    POLICIES_CONTEXT,
    build_model,
    build_parser,
    load_policies,
    normalize_session_id,
    require_question,
    run_chat,
    with_session_history,
)

SYSTEM_TEMPLATE = """\
Eres el asistente de atención al cliente de EventHub.
Respondes exclusivamente con la información contenida en las políticas de abajo.
Reglas:
- No uses conocimiento externo ni completes datos que no aparezcan en el texto.
- No inventes plazos, importes, excepciones ni estados de pedido.
- Si la pregunta no puede responderse con las políticas, di claramente que no dispones de esa información.
- El historial solo sirve para entender el hilo de la conversación. No aporta hechos nuevos.
- Responde en el mismo idioma que la pregunta, de forma breve y directa.
""" + POLICIES_CONTEXT

_cached_chain: Runnable | None = None


def build_prompt() -> ChatPromptTemplate:
    """Sistema, historial de la sesión y pregunta actual."""
    return ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_TEMPLATE),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{question}"),
        ]
    )


def build_chain(policies: str | None = None) -> Runnable:
    """Compone prompt | model | parser y lo envuelve con el historial."""
    policies_text = load_policies() if policies is None else policies
    prompt = build_prompt().partial(policies=policies_text)
    core_chain = prompt | build_model() | build_parser()
    return with_session_history(core_chain)


def answer_question(question: str, session_id: str) -> str:
    """Ejecuta la cadena del paso 2 en la sesión indicada."""
    global _cached_chain
    text = require_question(question)
    session = normalize_session_id(session_id)
    if _cached_chain is None:
        _cached_chain = build_chain()
    return _cached_chain.invoke(
        {"question": text},
        config={"configurable": {"session_id": session}},
    )


def main() -> None:
    run_chat(
        answer_question,
        use_session=True,
        title="Paso 2. Políticas con memoria. Sin herramientas.",
    )


if __name__ == "__main__":
    main()
