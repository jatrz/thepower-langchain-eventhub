"""Paso 3: memoria del paso 2 más buscar_eventos y consultar_pedido.

Ejecutar: python paso3.py
"""

from __future__ import annotations

from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import Runnable, RunnableLambda
from langchain_openai import ChatOpenAI

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
from tools.events_tool import buscar_eventos
from tools.orders_tool import consultar_pedido

MAX_TOOL_ROUNDS = 4
TOOLS = [buscar_eventos, consultar_pedido]
TOOLS_BY_NAME = {item.name: item for item in TOOLS}

SYSTEM_TEMPLATE = """\
Eres el asistente de atención al cliente de EventHub.
Responde en el mismo idioma que la pregunta, de forma breve y directa.
No inventes datos. Si no están en las políticas ni en el resultado de una herramienta, di que no dispones de esa información.

Fuentes permitidas:
- Políticas de abajo: devoluciones, cancelaciones, cambios, reembolsos y plazos. No completes normas que no aparezcan en el texto.
- Herramienta buscar_eventos: eventos, ciudad, categoría, fecha, precio y disponibilidad. Úsala antes de citar un evento. Puedes filtrar por ciudad, por categoría o por ambas. No inventes eventos ni plazas.
- Herramienta consultar_pedido: estado de un pedido. Necesita id_pedido y email. Si el usuario no repite el id pero ya está en el historial, reutilízalo. No adivines el email: usa solo el que el usuario haya escrito.
- Historial: sirve para entender el hilo, por ejemplo "el más barato de esos" o "los eventos de antes". No aporta hechos que no se hayan consultado ya.

Protección de pedidos:
- Llama a consultar_pedido antes de hablar de un pedido.
- Si la herramienta dice que el email no coincide o que el pedido no existe, transmite solo ese resultado.
- No reveles estado, eventos, cantidad ni el email registrado cuando la comprobación falle.
- No completes un email correcto que el usuario no haya dado.

Cuando hables de eventos, incluye nombre, precio y disponibilidad para que el historial conserve la consulta.
""" + POLICIES_CONTEXT

_cached_chain: Runnable | None = None


def build_prompt() -> ChatPromptTemplate:
    """Sistema con políticas, historial y pregunta."""
    return ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_TEMPLATE),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{question}"),
        ]
    )

def execute_tool_call(call: dict) -> ToolMessage:
    """Ejecuta una herramienta pedida por el modelo y devuelve su ToolMessage."""
    name = call.get("name") or ""
    tool = TOOLS_BY_NAME.get(name)
    if tool is None:
        content = f"La herramienta {name} no está disponible."
    else:
        try:
            result = tool.invoke(call.get("args") or {})
            content = result if isinstance(result, str) else str(result)
        except Exception as exc:
            content = f"La herramienta {name} no ha podido ejecutarse: {exc}"
    return ToolMessage(content=content, tool_call_id=call.get("id") or name, name=name)


def build_tool_runner(model: ChatOpenAI) -> RunnableLambda:
    """Llama al modelo y ejecuta las herramientas antes del parser."""
    bound_model = model.bind_tools(TOOLS)

    def run(prompt_value: object) -> AIMessage:
        messages = list(prompt_value.to_messages())
        ai_message = AIMessage(content="")
        for _ in range(MAX_TOOL_ROUNDS):
            ai_message = bound_model.invoke(messages)
            if not ai_message.tool_calls:
                return ai_message
            messages.append(ai_message)
            messages.extend(execute_tool_call(call) for call in ai_message.tool_calls)
        if not ai_message.content:
            return AIMessage(
                content="No he podido completar la consulta con las herramientas disponibles."
            )
        return ai_message

    return RunnableLambda(run)


def build_chain(policies: str | None = None) -> Runnable:
    """Compone prompt | modelo_con_herramientas | parser, con historial."""
    policies_text = load_policies() if policies is None else policies
    prompt = build_prompt().partial(policies=policies_text)
    core_chain = prompt | build_tool_runner(build_model()) | build_parser()
    return with_session_history(core_chain)


def answer_question(question: str, session_id: str) -> str:
    """Ejecuta la cadena del paso 3 en la sesión indicada."""
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
        title="Paso 3. Políticas, memoria y herramientas.",
    )


if __name__ == "__main__":
    main()
