"""Reserva de entradas con confirmación previa. No descuenta hasta el 'confirmo'."""

from __future__ import annotations

import json
import threading
import unicodedata
from pathlib import Path

from langchain_core.tools import tool

from tools.events_tool import EVENTS_PATH, load_events

_LOCK = threading.Lock()
_PENDING: dict[str, dict] = {}

CONFIRMATION_WORDS = {
    "si",
    "confirmo",
    "confirmar",
    "si confirmo",
    "confirmo la reserva",
    "si, confirmo",
}


def clear_pending_reservations() -> None:
    """Borra propuestas pendientes. Útil en tests."""
    _PENDING.clear()


def es_confirmacion_explicita(texto: str) -> bool:
    """Solo acepta una confirmación literal, no un 'ok' o un Enter vacío."""
    normalized = unicodedata.normalize("NFD", str(texto).strip().casefold())
    without_accents = "".join(
        char for char in normalized if unicodedata.category(char) != "Mn"
    )
    compact = " ".join(without_accents.replace("¡", "").replace("!", "").split())
    compact = compact.strip(" .,")
    return compact in CONFIRMATION_WORDS


def _parse_cantidad(value: object) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value) if value.is_integer() else None
    text = str(value).strip()
    if not text or any(char in text for char in ".eE"):
        try:
            number = int(text)
        except ValueError:
            return None
        return number
    try:
        return int(text)
    except ValueError:
        return None


def _find_event(events: list[dict], id_evento: str) -> dict | None:
    target = str(id_evento).strip().casefold()
    for event in events:
        if isinstance(event, dict) and str(event.get("id", "")).strip().casefold() == target:
            return event
    return None


def _save_events(events: list[dict], path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(events, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def reservar_entradas_controlada(
    id_evento: str,
    cantidad: object,
    *,
    session_id: str = "tool",
    ultimo_mensaje_usuario: str | None = None,
    mensajes_usuario: int = 0,
    path: Path = EVENTS_PATH,
) -> str:
    """Valida y solo descuenta si hay una confirmación posterior a la propuesta.

    La primera llamada deja una propuesta pendiente y no escribe events.json.
    Una segunda llamada, con el mismo evento y cantidad, descuenta solo si el
    último mensaje del usuario es una confirmación explícita y llegó después.
    """
    parsed = _parse_cantidad(cantidad)
    if parsed is None or parsed <= 0:
        return (
            "No se ha reservado nada. La cantidad debe ser un entero mayor que cero."
        )

    with _LOCK:
        events = load_events(path)
        event = _find_event(events, id_evento)
        if event is None:
            return f"No existe el evento {id_evento}. No se ha modificado la disponibilidad."

        disponible = event.get("disponibilidad")
        if not isinstance(disponible, int) or isinstance(disponible, bool):
            return (
                "No se ha reservado nada. El catálogo no tiene una disponibilidad válida "
                f"para {event.get('id')}."
            )
        if parsed > disponible:
            return (
                f"No hay disponibilidad suficiente para {event.get('nombre')} "
                f"({event.get('id')}). Quedan {disponible} entradas y se han pedido {parsed}. "
                "No se ha descontado nada."
            )

        key = str(session_id).strip() or "tool"
        pending = _PENDING.get(key)
        confirmed = (
            pending is not None
            and pending["id"] == event["id"]
            and pending["cantidad"] == parsed
            and mensajes_usuario > pending["mensajes_usuario"]
            and es_confirmacion_explicita(ultimo_mensaje_usuario or "")
        )
        if not confirmed:
            _PENDING[key] = {
                "id": event["id"],
                "cantidad": parsed,
                "mensajes_usuario": mensajes_usuario,
            }
            return (
                "Reserva NO realizada y disponibilidad sin cambios. "
                f"Hay {disponible} entradas para {event.get('nombre')} ({event.get('id')}). "
                f"Para reservar {parsed}, pide al usuario que responda exactamente confirmo "
                "en un mensaje posterior. No digas que la reserva ya está hecha."
            )

        event["disponibilidad"] = disponible - parsed
        _save_events(events, path)
        _PENDING.pop(key, None)
        return (
            "Reserva confirmada y guardada. "
            f"{event.get('nombre')} ({event.get('id')}): reservadas {parsed}; "
            f"disponibilidad restante {event['disponibilidad']}."
        )


@tool
def reservar_entradas(id_evento: str, cantidad: int) -> str:
    """Prepara una reserva de entradas. No descuenta plazas por sí sola.

    Úsala cuando el usuario quiera reservar entradas de un evento concreto.
    Necesita el id del evento, por ejemplo EVT-001, y una cantidad entera mayor
    que cero. No inventes ids ni disponibilidad.

    La primera llamada solo comprueba que el evento existe y que hay plazas.
    No guarda nada. Después debes pedir al usuario que responda exactamente
    "confirmo". Solo en un turno posterior, y solo si el usuario lo ha escrito,
    vuelve a llamar a esta herramienta con el mismo id_evento y la misma cantidad.
    Si el resultado dice que la reserva no está hecha, no afirmes lo contrario.

    Args:
        id_evento: Identificador real del evento, como EVT-001.
        cantidad: Número entero de entradas a reservar. Debe ser mayor que cero
            y no puede superar la disponibilidad.
    """
    return reservar_entradas_controlada(id_evento, cantidad)
