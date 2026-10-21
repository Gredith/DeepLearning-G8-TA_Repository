"""Borra de los notebooks las salidas que muestran caras de FaceForensics++.

Los términos de FF++ no permiten redistribuir videos, frames ni recortes de caras. Las celdas que
los muestran llevan el tag `caras`; este script vacía sus salidas antes de versionar un notebook
ejecutado en Kaggle.

Uso (desde la raíz del repositorio):
    python tools/limpiar_caras.py            # limpia notebooks/**/*.ipynb en el lugar
    python tools/limpiar_caras.py --check    # solo verifica; sale con 1 si encuentra problemas
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

TAG = "caras"
# El script de descarga solo se entrega tras la aprobación de acceso; su URL tampoco se publica.
URL_PROHIBIDA = "kaldir.vc.in.tum.de/faceforensics_download"


def _tiene_tag(cell: dict) -> bool:
    return TAG in cell.get("metadata", {}).get("tags", [])


def revisar(path: Path, limpiar: bool) -> list[str]:
    """Devuelve los problemas encontrados en un notebook; si `limpiar`, borra las salidas marcadas."""
    texto = path.read_text(encoding="utf-8")
    notebook = json.loads(texto)
    problemas = []
    modificado = False
    for index, cell in enumerate(notebook.get("cells", [])):
        if _tiene_tag(cell) and cell.get("outputs"):
            if limpiar:
                cell["outputs"] = []
                cell["execution_count"] = None
                modificado = True
            else:
                problemas.append(f"{path}: celda {index} (tag '{TAG}') tiene salidas")
    if URL_PROHIBIDA in texto:
        problemas.append(f"{path}: contiene la URL del script de descarga de FF++ (bórrala a mano)")
    if modificado:
        path.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"limpio: {path}")
    return problemas


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true", help="solo verificar, sin modificar archivos")
    parser.add_argument("paths", nargs="*", type=Path, help="notebooks a revisar (por defecto notebooks/**/*.ipynb)")
    args = parser.parse_args()
    notebooks = args.paths or sorted(Path("notebooks").rglob("*.ipynb"))
    notebooks = [path for path in notebooks if ".ipynb_checkpoints" not in path.parts]
    if not notebooks:
        print("No se encontraron notebooks; ejecutar desde la raíz del repositorio.", file=sys.stderr)
        return 1
    problemas = [p for path in notebooks for p in revisar(path, limpiar=not args.check)]
    for problema in problemas:
        print(problema, file=sys.stderr)
    if not problemas:
        print(f"OK: {len(notebooks)} notebooks sin salidas de caras ni URL del script.")
    return 1 if problemas else 0


if __name__ == "__main__":
    sys.exit(main())
