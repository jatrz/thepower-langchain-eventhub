"""Tests de los tres pasos. No llaman al modelo."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableSequence
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI

import common
import paso1
import paso2
import paso3
import paso3_entradas
from common import clear_sessions, get_session_history, run_chat
from tools.events_tool import buscar_eventos
from tools.orders_tool import consultar_pedido, load_orders
from tools.reservations_tool import (
    clear_pending_reservations,
    reservar_entradas,
    reservar_entradas_controlada,
)


class CommonTests(unittest.TestCase):
    def test_load_policies_incluye_devoluciones(self) -> None:
        text = common.load_policies()
        self.assertIn("7 días naturales", text)
        self.assertIn("comisión de gestión de 2 EUR", text)

    def test_sesiones_independientes_y_limite_de_diez(self) -> None:
        clear_sessions()
        first = get_session_history("a")
        second = get_session_history("b")
        self.assertIs(first, get_session_history(" a "))
        self.assertIsNot(first, second)

        for index in range(8):
            first.add_messages(
                [
                    HumanMessage(content=f"pregunta-{index}"),
                    AIMessage(content=f"respuesta-{index}"),
                ]
            )

        self.assertEqual(len(first.messages), 10)
        self.assertEqual(first.messages[0].content, "pregunta-3")
        self.assertEqual(second.messages, [])

    def test_run_chat_sale_sin_llamar_al_modelo(self) -> None:
        calls: list[str] = []

        def answer(question: str) -> str:
            calls.append(question)
            return "no debería usarse"

        inputs = iter(["", "salir"])
        lines: list[str] = []
        run_chat(
            answer,
            use_session=False,
            title="prueba",
            input_fn=lambda _prompt="": next(inputs),
            output_fn=lambda *args: lines.append(" ".join(str(arg) for arg in args)),
        )
        self.assertEqual(calls, [])
        self.assertIn("Hasta luego.", lines)


class Paso1Tests(unittest.TestCase):
    def test_prompt_solo_politicas_y_pregunta(self) -> None:
        prompt = paso1.build_prompt().partial(policies="POLITICA DE PRUEBA")
        text = prompt.invoke({"question": "¿Puedo devolver?"}).to_string()
        self.assertIn("POLITICA DE PRUEBA", text)
        self.assertIn("¿Puedo devolver?", text)
        self.assertIn("exclusivamente", text)
        self.assertNotIn("buscar_eventos", paso1.SYSTEM_TEMPLATE)

    def test_cadena_es_prompt_modelo_parser(self) -> None:
        chain = paso1.build_chain(policies="solo prueba")
        self.assertIsInstance(chain, RunnableSequence)
        self.assertNotIsInstance(chain, RunnableWithMessageHistory)
        self.assertIsInstance(chain.steps[1], ChatOpenAI)

    def test_pregunta_vacia(self) -> None:
        with self.assertRaises(ValueError):
            paso1.answer_question("   ")


class Paso2Tests(unittest.TestCase):
    def test_prompt_incluye_historial(self) -> None:
        prompt = paso2.build_prompt().partial(policies="POLITICA DE PRUEBA")
        text = prompt.invoke(
            {
                "question": "¿Y la comisión?",
                "history": [HumanMessage(content="Quiero devolver")],
            }
        ).to_string()
        self.assertIn("POLITICA DE PRUEBA", text)
        self.assertIn("Quiero devolver", text)
        self.assertIn("¿Y la comisión?", text)
        self.assertNotIn("buscar_eventos", paso2.SYSTEM_TEMPLATE)

    def test_cadena_tiene_historial_por_sesion(self) -> None:
        chain = paso2.build_chain(policies="solo prueba")
        self.assertIsInstance(chain, RunnableWithMessageHistory)

    def test_pregunta_o_sesion_vacias(self) -> None:
        with self.assertRaises(ValueError):
            paso2.answer_question("", "sesion")
        with self.assertRaises(ValueError):
            paso2.answer_question("hola", "  ")


class Paso3Tests(unittest.TestCase):
    def test_herramientas_conectadas(self) -> None:
        self.assertEqual(
            [tool.name for tool in paso3.TOOLS],
            ["buscar_eventos", "consultar_pedido"],
        )
        chain = paso3.build_chain(policies="solo prueba")
        self.assertIsInstance(chain, RunnableWithMessageHistory)
        self.assertIn("buscar_eventos", paso3.SYSTEM_TEMPLATE)
        self.assertIn("consultar_pedido", paso3.SYSTEM_TEMPLATE)
        self.assertNotIn("reservar_entradas", paso3.SYSTEM_TEMPLATE)
        self.assertEqual(
            [tool.name for tool in paso3_entradas.TOOLS],
            ["buscar_eventos", "consultar_pedido", "reservar_entradas"],
        )
        self.assertIn("reservar_entradas", paso3_entradas.SYSTEM_TEMPLATE)

    def test_buscar_eventos_filtra_y_no_inventa(self) -> None:
        found = buscar_eventos.invoke({"ciudad": "Valencia", "categoria": "concierto"})
        self.assertIn("Rock en la Ribera", found)
        self.assertIn("precio:", found)
        self.assertIn("disponibilidad:", found)
        self.assertNotIn("Feria del Libro", found)

        empty = buscar_eventos.invoke({"ciudad": "Bilbao", "categoria": "concierto"})
        self.assertIn("No hay eventos", empty)

    def test_consultar_pedido_no_revela_si_el_email_falla(self) -> None:
        order = next(item for item in load_orders() if item["id_pedido"] == "PED-1001")
        denied = consultar_pedido.invoke(
            {"id_pedido": order["id_pedido"], "email": "incorrecto@example.com"}
        )
        self.assertIn("no coincide", denied.casefold())
        for secret in (order["email"], order["estado"], "EVT-001", "cantidad"):
            self.assertNotIn(str(secret).casefold(), denied.casefold())

        missing = consultar_pedido.invoke(
            {"id_pedido": "ORD-1001", "email": order["email"]}
        )
        self.assertIn("No existe", missing)
        self.assertNotIn(order["email"], missing)

    def test_email_correcto_devuelve_estado(self) -> None:
        order = next(item for item in load_orders() if item["id_pedido"] == "PED-1001")
        allowed = consultar_pedido.invoke(
            {"id_pedido": order["id_pedido"], "email": order["email"]}
        )
        self.assertIn(f"estado: {order['estado']}", allowed)
        self.assertIn("EVT-001", allowed)
        self.assertNotIn(order["email"], allowed)

    def test_execute_tool_call_sin_modelo(self) -> None:
        message = paso3.execute_tool_call(
            {
                "name": "buscar_eventos",
                "id": "call-1",
                "args": {"ciudad": "Sevilla", "categoria": ""},
            }
        )
        self.assertIn("Taller de Cocina Mediterránea", message.content)
        self.assertNotIn("Valencia Jazz", message.content)


class ReservaTests(unittest.TestCase):
    def setUp(self) -> None:
        clear_pending_reservations()
        self.catalog = [
            {
                "id": "EVT-001",
                "nombre": "Noche Indie en la Marina",
                "disponibilidad": 3,
            }
        ]
        self.directory = tempfile.TemporaryDirectory()
        self.path = Path(self.directory.name) / "events.json"
        self.path.write_text(
            json.dumps(self.catalog, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def tearDown(self) -> None:
        clear_pending_reservations()
        self.directory.cleanup()

    def _disponibilidad(self) -> int:
        data = json.loads(self.path.read_text(encoding="utf-8"))
        return data[0]["disponibilidad"]

    def test_no_descuenta_sin_confirmacion_posterior(self) -> None:
        preview = reservar_entradas_controlada(
            "EVT-001",
            2,
            session_id="s1",
            ultimo_mensaje_usuario="Reserva 2 entradas",
            mensajes_usuario=1,
            path=self.path,
        )
        self.assertIn("NO realizada", preview)
        self.assertEqual(self._disponibilidad(), 3)

        same_turn = reservar_entradas_controlada(
            "EVT-001",
            2,
            session_id="s1",
            ultimo_mensaje_usuario="confirmo",
            mensajes_usuario=1,
            path=self.path,
        )
        self.assertIn("NO realizada", same_turn)
        self.assertEqual(self._disponibilidad(), 3)

    def test_confirmo_posterior_descuenta_y_guarda(self) -> None:
        reservar_entradas_controlada(
            "EVT-001",
            2,
            session_id="s1",
            ultimo_mensaje_usuario="Quiero 2",
            mensajes_usuario=1,
            path=self.path,
        )
        done = reservar_entradas_controlada(
            "evt-001",
            2,
            session_id="s1",
            ultimo_mensaje_usuario="Sí, confirmo",
            mensajes_usuario=2,
            path=self.path,
        )
        self.assertIn("Reserva confirmada", done)
        self.assertEqual(self._disponibilidad(), 1)

    def test_otra_sesion_no_confirma_la_propuesta(self) -> None:
        reservar_entradas_controlada(
            "EVT-001",
            2,
            session_id="s1",
            ultimo_mensaje_usuario="Quiero 2",
            mensajes_usuario=1,
            path=self.path,
        )
        other = reservar_entradas_controlada(
            "EVT-001",
            2,
            session_id="s2",
            ultimo_mensaje_usuario="confirmo",
            mensajes_usuario=2,
            path=self.path,
        )
        self.assertIn("NO realizada", other)
        self.assertEqual(self._disponibilidad(), 3)

    def test_rechaza_cantidad_invalida_o_superior(self) -> None:
        for cantidad in (0, -1, "2.5", "cero"):
            reply = reservar_entradas_controlada(
                "EVT-001",
                cantidad,
                session_id="s1",
                ultimo_mensaje_usuario="confirmo",
                mensajes_usuario=2,
                path=self.path,
            )
            self.assertNotIn("Reserva confirmada", reply)
        too_many = reservar_entradas_controlada(
            "EVT-001",
            4,
            session_id="s1",
            ultimo_mensaje_usuario="confirmo",
            mensajes_usuario=3,
            path=self.path,
        )
        self.assertIn("No hay disponibilidad", too_many)
        missing = reservar_entradas_controlada(
            "EVT-999",
            1,
            session_id="s1",
            ultimo_mensaje_usuario="confirmo",
            mensajes_usuario=3,
            path=self.path,
        )
        self.assertIn("No existe", missing)
        self.assertEqual(self._disponibilidad(), 3)

    def test_la_herramienta_directa_no_escribe_el_catalogo_real(self) -> None:
        reply = reservar_entradas.invoke({"id_evento": "EVT-001", "cantidad": 1})
        self.assertIn("NO realizada", reply)


class AppTests(unittest.TestCase):
    def test_app_rechaza_un_paso_desconocido(self) -> None:
        import app

        with patch("sys.argv", ["app.py", "9"]):
            with self.assertRaises(SystemExit) as raised:
                app.main()
        self.assertEqual(raised.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
