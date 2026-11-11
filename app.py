"""Lanza el paso que quieras probar: python app.py 1|2|3"""

from __future__ import annotations

import sys

from paso1 import main as main_paso1
from paso2 import main as main_paso2
from paso3 import main as main_paso3

STEPS = {
    "1": main_paso1,
    "2": main_paso2,
    "3": main_paso3,
}


def main() -> None:
    step = sys.argv[1] if len(sys.argv) > 1 else "3"
    runner = STEPS.get(step)
    if runner is None:
        print("Uso: python app.py [1|2|3]")
        print("  1  políticas, sin memoria ni herramientas")
        print("  2  políticas con memoria")
        print("  3  políticas, memoria y herramientas")
        raise SystemExit(2)
    runner()


if __name__ == "__main__":
    main()
