"""Código compartido por los tres pasos de EventHub."""

from __future__ import annotations

import os
import warnings
from collections.abc import Callable, Sequence
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI

PROJECT_ROOT = Path(__file__).resolve().parent
POLICIES_PATH = PROJECT_ROOT / "data" / "policies.md"

load_dotenv(PROJECT_ROOT / ".env")

MAX_HISTORY_MESSAGES = 10
EXIT_COMMAND = "salir"

POLICIES_CONTEXT = """
Políticas de EventHub:
{policies}
"""


def load_policies(path: Path = POLICIES_PATH) -> str:
    """Lee el texto completo de las políticas."""
    if not path.is_file():
        raise FileNotFoundError(f"No se encuentra el fichero de políticas: {path}")
    return path.read_text(encoding="utf-8").strip()


def build_model() -> ChatOpenAI:
    """Cliente ChatOpenAI. Usa OPENAI_MODEL si está definido."""
    model_name = os.getenv("OPENAI_MODEL")
    kwargs: dict = {"temperature": 1}
    if model_name:
        kwargs["model"] = model_name
    return ChatOpenAI(**kwargs)


def build_parser() -> StrOutputParser:
    """Convierte el AIMessage final en texto."""
    return StrOutputParser()


def require_question(question: str) -> str:
    if not question or not str(question).strip():
        raise ValueError("La pregunta no puede estar vacía.")
    return str(question).strip()


def normalize_session_id(session_id: str) -> str:
    if session_id is None or not str(session_id).strip():
        raise ValueError("session_id no puede estar vacío.")
    return str(session_id).strip()


class SessionChatHistory(BaseChatMessageHistory):
    """Historial en memoria de una sola sesión, recortado a 10 mensajes."""

    def __init__(self) -> None:
        super().__init__()
        self._messages: list[BaseMessage] = []

    @property
    def messages(self) -> list[BaseMessage]:
        return list(self._messages[-MAX_HISTORY_MESSAGES:])

    def add_messages(self, messages: Sequence[BaseMessage]) -> None:
        self._messages.extend(messages)
        if len(self._messages) > MAX_HISTORY_MESSAGES:
            del self._messages[:-MAX_HISTORY_MESSAGES]

    def clear(self) -> None:
        self._messages.clear()


_SESSION_STORE: dict[str, SessionChatHistory] = {}


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    """Devuelve el historial de session_id. Dos ids no comparten objeto."""
    key = normalize_session_id(session_id)
    history = _SESSION_STORE.get(key)
    if history is None:
        history = SessionChatHistory()
        _SESSION_STORE[key] = history
    return history


def clear_sessions() -> None:
    """Vacía el almacén de sesiones. Útil en tests."""
    _SESSION_STORE.clear()


def with_session_history(chain: Runnable) -> Runnable:
    """Envuelve una cadena para leer y guardar el historial por session_id."""
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="RunnableWithMessageHistory is deprecated",
        )
        return RunnableWithMessageHistory(
            chain,
            get_session_history,
            input_messages_key="question",
            history_messages_key="history",
        )


def run_chat(
    answer: Callable[..., str],
    *,
    use_session: bool,
    title: str,
    session_id: str = "consola",
    input_fn: Callable[[str], str] = input,
    output_fn: Callable[..., None] = print,
) -> None:
    """Bucle de terminal. Con use_session=False llama answer(pregunta)."""
    output_fn(title)
    output_fn("Escribe una pregunta. Escribe 'salir' para terminar.")
    if use_session:
        output_fn(f"Sesión: {session_id}")

    while True:
        try:
            question = input_fn("\nTú: ").strip()
        except (EOFError, KeyboardInterrupt):
            output_fn("\nHasta luego.")
            break

        if not question:
            continue
        if question.lower() == EXIT_COMMAND:
            output_fn("Hasta luego.")
            break

        try:
            if use_session:
                reply = answer(question, session_id=session_id)
            else:
                reply = answer(question)
        except Exception as exc:
            output_fn(f"No se ha podido obtener la respuesta: {exc}")
            continue

        output_fn(f"EventHub: {reply}")
