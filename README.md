# thepower-langchain-eventhub

Asistente de atención al cliente para un catálogo ficticio de eventos. Responde con las políticas del proyecto, recuerda la conversación de una sesión y, a partir del paso 3, consulta eventos y pedidos reales guardados en JSON.

Los datos de `data/` son de práctica. No son pedidos ni clientes reales.

## Objetivo académico

El proyecto se construye por pasos con LangChain y LCEL:

1. Responder solo con `data/policies.md`.
2. Añadir memoria independiente por `session_id`.
3. Conectar las herramientas `buscar_eventos` y `consultar_pedido` sin inventar datos ni revelar un pedido si el email no coincide.

Cada paso se puede ejecutar por separado. El paso 3 no incluye la reserva de entradas. Esa mejora opcional está solo en `paso3_entradas.py`.

## Tecnologías utilizadas

- Python 3, en el entorno virtual de WSL (`.venv/bin/python`).
- LangChain 1.4.9 y LCEL (`prompt | model | parser`).
- `langchain-openai` 1.7.4 y `openai` 3.46.0.
- `python-dotenv` 1.2.4 para leer `.env`.
- `unittest` de la biblioteca estándar. No hace falta instalar pytest.

Las versiones fijadas están en `requirements.txt`.

## Estructura de carpetas

```text
thepower-langchain-eventhub/
├── app.py                 # python app.py 1|2|3
├── common.py              # políticas, modelo, parser, memoria y consola
├── paso1.py               # políticas, sin memoria ni herramientas
├── paso2.py               # políticas con memoria
├── paso3.py               # políticas, memoria, eventos y pedidos
├── paso3_entradas.py      # mejora opcional: reserva con confirmación
├── assistant/chain.py     # compatibilidad: reexporta el paso 3
├── data/
│   ├── policies.md
│   ├── events.json
│   └── orders.json
├── tools/
│   ├── events_tool.py
│   ├── orders_tool.py
│   └── reservations_tool.py
├── tests/test_pasos.py
├── requirements.txt
├── .env.example
└── .gitignore
```

`assistant/chain.py` apunta al paso 3, no a la reserva.

## LCEL: prompt, model y parser

LCEL compone piezas ejecutables con el operador `|`. La cadena mínima del paso 1 es:

```text
prompt | model | parser
```

- **Prompt.** `ChatPromptTemplate` arma los mensajes. El mensaje de sistema incluye el texto de `data/policies.md`. El mensaje humano es la pregunta. Las políticas se fijan con `prompt.partial(policies=...)` antes del pipe, porque en esta versión `RunnableSequence` no tiene `.partial()`.
- **Model.** `ChatOpenAI` con `temperature` 1. Si `OPENAI_MODEL` está definido, se usa ese modelo. Si no, se usa el modelo por defecto del cliente.
- **Parser.** `StrOutputParser` convierte el mensaje final del modelo en un `str`.

En el paso 2 el prompt añade `MessagesPlaceholder("history")` y la cadena se envuelve con `RunnableWithMessageHistory`. La composición visible sigue siendo `prompt | model | parser`.

En el paso 3 el modelo va dentro de un `RunnableLambda` que puede llamar a las herramientas antes de devolver el mensaje final. La cadena queda `prompt | tool_runner | parser`, también envuelta con el historial.

## Memoria y session_id

La memoria vive en el proceso, en el diccionario `_SESSION_STORE` de `common.py`. La clave es el `session_id` sin espacios alrededor. `"sesion-a"` y `" sesion-a "` son la misma sesión. Dos ids distintos no comparten mensajes.

`RunnableWithMessageHistory` pide el historial a `get_session_history(session_id)` antes de llamar al modelo y guarda después la pregunta y la respuesta final. No guarda los mensajes intermedios de herramientas. El historial se recorta a los últimos 10 mensajes.

La consola de los pasos 2 y 3 usa la sesión `consola`. Para salir hay que escribir `salir`. Una pregunta vacía no se envía al modelo.

El almacén se borra al cerrar el proceso. No hay base de datos de conversaciones.

## Herramientas

### buscar_eventos

`buscar_eventos(ciudad, categoria)` lee `data/events.json`. Se puede filtrar por ciudad, por categoría o por ambas. Un argumento vacío no filtra ese campo. Si ambos están vacíos, pide al menos un filtro.

La comparación ignora mayúsculas y acentos, pero exige coincidencia exacta del valor normalizado. Devuelve nombre, ciudad, categoría, fecha, precio y disponibilidad. Si un campo no está en el JSON, lo indica como no disponible. Si no hay coincidencias, lo dice. No inventa eventos.

### consultar_pedido

`consultar_pedido(id_pedido, email)` lee `data/orders.json`. Busca el `id_pedido` sin distinguir mayúsculas. El email tiene que coincidir con el email registrado, también sin distinguir mayúsculas.

Si coincide, devuelve `id_pedido`, `estado`, `eventos` y `cantidad`. No devuelve el email. Si no coincide, o si el pedido no existe, no muestra estado, eventos, cantidad ni el email registrado.

## Instalación y entorno virtual

Sitúate en la carpeta del proyecto `thepower-langchain-eventhub` dentro de Ubuntu:

```bash
cd thepower-langchain-eventhub
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Si `.venv` ya existe, basta con activarlo:

```bash
cd thepower-langchain-eventhub
source .venv/bin/activate
```

El intérprete de Windows `.venv/Scripts/python.exe` no forma parte de este entorno. Hay que usar `.venv/bin/python` dentro de WSL.

## Configuración de .env

```bash
cp .env.example .env
```

`.env.example` contiene:

```text
OPENAI_API_KEY=pon_aqui_tu_api_key
OPENAI_MODEL=gpt-6-luna
```

Sustituye `OPENAI_API_KEY` por tu clave. No la subas al repositorio: `.env` está en `.gitignore`. `common.py` carga `.env` al importarse. `OPENAI_MODEL` es opcional. Si lo quitas, se usa el modelo por defecto de `ChatOpenAI`.

## Cómo ejecutar el proyecto

Con el entorno activado y el directorio de trabajo en la raíz:

```bash
python paso1.py
python paso2.py
python paso3.py
python app.py 1
python app.py 2
python app.py 3
python app.py
```

`python app.py` sin argumento ejecuta el paso 3. Un paso desconocido muestra el uso y termina con código 2:

```bash
python app.py 9
```

La mejora opcional de reservas no está en `app.py`. Se lanza aparte:

```bash
python paso3_entradas.py
```

Escribe `salir` para cerrar la consola.

## Pruebas de los tres pasos

Los tests están en `tests/test_pasos.py` y no llaman al modelo. Comprueban políticas, aislamiento de sesiones, el límite de 10 mensajes, el tipo de cadena de cada paso, el filtrado de eventos, la privacidad de pedidos y que la reserva no escribe el catálogo antes de una confirmación posterior.

Hay que ejecutarlos desde la raíz para que los imports encuentren los módulos:

```bash
source .venv/bin/activate
PYTHONPATH=. python -m unittest discover -s tests -v
```

No hay un script de conversación en vivo en `tests/`. La conversación de seis turnos se prueba a mano con el paso 3.

## Conversación obligatoria de seis turnos

Ejecuta el paso 3 y mantén la misma sesión. La consola ya usa `consola`. Si llamas a `answer_question` desde código, usa un único `session_id`, por ejemplo `prueba-eventhub-001`.

Los identificadores salen de `data/orders.json`. El pedido de ejemplo es `PED-1001`, con el email `ana.ejemplo@correo-ficticio.test`. No uses `ORD-1001`: no está en el fichero.

1. ¿Cuál es la política de devolución de entradas?
2. ¿Qué conciertos hay en Valencia?
3. ¿Cuál es el más barato de esos?
4. Consulta el pedido PED-1001 con el email incorrecto@example.com.
5. Ahora consúltalo con ana.ejemplo@correo-ficticio.test.
6. ¿Qué eventos habíamos encontrado antes?

Qué debe poder comprobarse con los ficheros, no con una respuesta inventada:

- El turno 1 solo puede usar `data/policies.md`: devolución hasta 7 días naturales antes, entrada sin usar, pedido confirmado y comisión de gestión de 2 EUR. No hay devolución el mismo día ni después del inicio.
- El turno 2 debe salir de `data/events.json`. En Valencia y categoría `concierto` están EVT-001, EVT-002, EVT-003, EVT-004 y EVT-005.
- El turno 3 depende del historial. El concierto de Valencia más barato en el JSON es Rock en la Ribera (EVT-005), a 22.0.
- El turno 4 no debe mostrar estado, eventos, cantidad ni el email registrado.
- El turno 5, con el email real, puede mostrar el estado `confirmado`, cantidad 2 y los eventos EVT-001 y EVT-003.
- El turno 6 debe recordar los eventos consultados en esa sesión. Si se abre otro `session_id`, no debe heredarlos.

La redacción exacta depende del modelo. Contrasta la respuesta con `data/policies.md`, `data/events.json` y `data/orders.json`.

## Seguridad de consulta de pedidos

- No se consulta un pedido sin `id_pedido` y email.
- El email se compara con el registrado y no se devuelve en la respuesta.
- Un email distinto produce un aviso de que no coincide. No incluye el estado ni datos del pedido.
- Un id ausente produce un aviso de que no existe. No sugiere otro pedido.
- El modelo tiene instrucción de no completar un email que el usuario no haya escrito. "El email correcto" no autoriza a mostrar el pedido.
- `assistant/chain.py` y el paso 3 usan esta herramienta. El paso 1 y el paso 2 no pueden consultar pedidos.

## Limitaciones y mejoras futuras

- La memoria desaparece al cerrar el programa y no se comparte entre procesos.
- El historial guarda solo la pregunta y la respuesta final, hasta 10 mensajes.
- El catálogo y los pedidos son JSON locales. No hay API ni autenticación de usuarios.
- `buscar_eventos` no hace búsqueda parcial ni por nombre. Filtra por ciudad y categoría.
- La temperatura del modelo es 1, así que dos ejecuciones pueden redactar distinto. Los datos no deben variar respecto de los ficheros.
- `RunnableWithMessageHistory` está obsoleto en LangChain reciente. Se mantiene porque el ejercicio lo pide.
- La reserva `reservar_entradas(id_evento, cantidad)` ya existe como mejora opcional en `paso3_entradas.py`. No descuenta plazas hasta que el usuario confirma en un mensaje posterior con `confirmo`, `sí` o `sí, confirmo`. No está conectada a `app.py` ni al paso 3.
- Posibles mejoras: persistir sesiones, confirmar la reserva sin depender de que el modelo vuelva a llamar a la herramienta, y añadir una forma de deshacer una reserva de prueba en `data/events.json`.
