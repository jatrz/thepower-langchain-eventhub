"""Herramienta de consulta de eventos, conectada al asistente."""

from __future__ import annotations

import json
import unicodedata
from pathlib import Path

from langchain_core.tools import tool

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EVENTS_PATH = PROJECT_ROOT / "data" / "events.json"

# Campos que puede devolver la herramienta. No se añaden datos que no estén en el JSON.
RESULT_FIELDS = (
    "nombre",
    "ciudad",
    "categoria",
    "fecha",
    "precio",
    "disponibilidad",
)


def load_events(path: Path = EVENTS_PATH) -> list[dict]:
    """Lee el catálogo real de eventos. No completa ni inventa registros."""
    if not path.is_file():
        raise FileNotFoundError(f"No se encuentra el catálogo de eventos: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("data/events.json debe contener una lista de eventos.")
    return data


def _normalize(value: object) -> str:
    text = str(value).strip().casefold()
    without_accents = unicodedata.normalize("NFD", text)
    return "".join(
        char for char in without_accents if unicodedata.category(char) != "Mn"
    )


def _has_filter(value: str) -> bool:
    return bool(value and value.strip())


def _matches(event: dict, ciudad: str, categoria: str) -> bool:
    if _has_filter(ciudad) and _normalize(event.get("ciudad", "")) != _normalize(ciudad):
        return False
    if _has_filter(categoria) and _normalize(event.get("categoria", "")) != _normalize(
        categoria
    ):
        return False
    return True


def _format_event(event: dict) -> str:
    parts: list[str] = []
    for field in RESULT_FIELDS:
        value = event.get(field)
        if value is None or value == "":
            shown = "dato no disponible en el catálogo"
        else:
            shown = value
        label = "categoría" if field == "categoria" else field
        parts.append(f"{label}: {shown}")
    return "- " + "; ".join(parts)


def _filter_label(ciudad: str, categoria: str) -> str:
    filters: list[str] = []
    if _has_filter(ciudad):
        filters.append(f"ciudad={ciudad.strip()}")
    if _has_filter(categoria):
        filters.append(f"categoría={categoria.strip()}")
    return ", ".join(filters)


@tool
def buscar_eventos(ciudad: str, categoria: str) -> str:
    """Busca eventos reales del catálogo de EventHub.

    Úsala cuando el usuario pregunte qué eventos hay, en qué ciudad se celebran,
    de qué categoría son, o quiera saber su fecha, precio o disponibilidad.
    No la uses para devoluciones, cancelaciones, cambios ni reembolsos: eso no
    está en el catálogo de eventos.

    Filtra por ciudad, por categoría o por ambos a la vez. Si indicas los dos,
    el evento debe coincidir con los dos. Si un parámetro llega vacío, no se
    filtra por ese campo. No inventes eventos ni plazas: solo puedes responder
    con lo que devuelva esta herramienta.

    Args:
        ciudad: Ciudad del evento, por ejemplo Valencia, Madrid, Barcelona o Sevilla.
            Cadena vacía si no se quiere filtrar por ciudad.
        categoria: Categoría del evento, por ejemplo concierto, cultura, deporte
            o gastronomia. Cadena vacía si no se quiere filtrar por categoría.
    """
    if not _has_filter(ciudad) and not _has_filter(categoria):
        return (
            "No se ha indicado ciudad ni categoría. "
            "Indica al menos uno de los dos filtros para buscar eventos."
        )

    matches = [
        event
        for event in load_events()
        if isinstance(event, dict) and _matches(event, ciudad, categoria)
    ]
    label = _filter_label(ciudad, categoria)
    if not matches:
        return f"No hay eventos que coincidan con {label}."

    lines = [f"Eventos encontrados ({len(matches)}) para {label}:"]
    lines.extend(_format_event(event) for event in matches)
    return "\n".join(lines)
