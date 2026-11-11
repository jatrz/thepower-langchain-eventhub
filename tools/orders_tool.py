"""Herramienta de consulta de pedidos, conectada al asistente."""

from __future__ import annotations

import json
from pathlib import Path

from langchain_core.tools import tool

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ORDERS_PATH = PROJECT_ROOT / "data" / "orders.json"

# Datos que se pueden mostrar solo después de verificar el email.
# El email registrado no se devuelve: sirve únicamente para la comprobación.
ALLOWED_FIELDS = ("id_pedido", "estado", "eventos", "cantidad")


def load_orders(path: Path = ORDERS_PATH) -> list[dict]:
    """Lee los pedidos reales. No completa ni inventa registros."""
    if not path.is_file():
        raise FileNotFoundError(f"No se encuentra el fichero de pedidos: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("data/orders.json debe contener una lista de pedidos.")
    return data


def _normalize_id(value: object) -> str:
    return str(value).strip().casefold()


def _normalize_email(value: object) -> str:
    return str(value).strip().casefold()


def _find_order(id_pedido: str) -> dict | None:
    target = _normalize_id(id_pedido)
    for order in load_orders():
        if isinstance(order, dict) and _normalize_id(order.get("id_pedido", "")) == target:
            return order
    return None


def _format_allowed(order: dict) -> str:
    lines = ["Pedido verificado. Datos disponibles:"]
    for field in ALLOWED_FIELDS:
        value = order.get(field)
        if value is None or value == "":
            shown = "dato no disponible"
        elif isinstance(value, list):
            shown = ", ".join(str(item) for item in value) if value else "dato no disponible"
        else:
            shown = value
        lines.append(f"- {field}: {shown}")
    return "\n".join(lines)


@tool
def consultar_pedido(id_pedido: str, email: str) -> str:
    """Consulta el estado de un pedido de EventHub si el email coincide.

    Úsala solo cuando el usuario quiera conocer el estado de su pedido y
    aporte el id_pedido y el email asociado a esa compra. No la uses para
    buscar eventos ni para explicar políticas de devolución o cancelación.

    Comprueba el email antes de mostrar nada. Si no coincide, no reveles el
    email registrado ni ningún otro dato del pedido. No inventes estados,
    eventos ni cantidades: responde solo con lo que devuelva esta herramienta.

    Args:
        id_pedido: Identificador del pedido, por ejemplo PED-1001.
        email: Correo que el usuario dice que está asociado al pedido.
            Debe coincidir con el email registrado para poder ver los datos.
    """
    if not str(id_pedido).strip() or not str(email).strip():
        return "Faltan datos. Indica el id_pedido y el email asociado al pedido."

    order = _find_order(id_pedido)
    if order is None:
        return "No existe ningún pedido con ese id_pedido."

    registered = order.get("email")
    if not isinstance(registered, str) or not registered.strip():
        return (
            "No se ha podido verificar el email del pedido. "
            "No se muestra ningún dato."
        )
    if _normalize_email(email) != _normalize_email(registered):
        return (
            "El email no coincide con el pedido indicado. "
            "No se puede mostrar ningún dato del pedido."
        )

    return _format_allowed(order)
