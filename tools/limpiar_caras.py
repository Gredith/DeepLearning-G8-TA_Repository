"""Borra de los notebooks las salidas que muestran caras de FaceForensics++ o el log del script de descarga.

Los términos de FF++ no permiten redistribuir videos, frames ni recortes de caras, ni el script oficial
de descarga. Kaggle sincroniza los notebooks con GitHub **con sus salidas**, así que este script es la
segunda barrera (la primera es que los notebooks no muestran caras: ver `kaggle_io.save_private`).

Se borran:
1. Las imágenes y el HTML con imágenes incrustadas de celdas con el tag `caras` (su texto, p. ej. «figura privada guardada…», y las tablas se conservan).
2. Las imágenes (`image/*`) de celdas cuyo código toca recortes, frames o Grad-CAM.
3. Las salidas de texto que contienen el log o la ruta del script de descarga de FF++.

Uso (desde la raíz del repositorio):
    python tools/limpiar_caras.py            # limpia notebooks/**/*.ipynb en el lugar
    python tools/limpiar_caras.py --check    # solo verifica; sale con 1 si encuentra problemas
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

TAG = "caras"
# El script de descarga solo se entrega tras la aprobación de acceso; ni su log, ni su ruta, ni su URL se publican.
URL_PROHIBIDA = "kaldir.vc.in.tum.de/faceforensics_download"
TEXTO_PROHIBIDO = re.compile(
    r"kaldir\.vc\.in\.tum\.de|FaceForensics_TOS|Downloading videos of dataset|faceforensics_download|"
    r"download[-_]faceforensics", re.IGNORECASE)
# Código que produce imágenes con caras del dataset
CODIGO_CON_CARAS = re.compile(
    r"CROPS|crops|gradcam\.|GradCAM\(|imshow\(\s*Image\.open|frames_por_dataset|extraer_frames|denorm\(|"
    r"intermediate_activations|VideoCapture", re.IGNORECASE)
# HTML que puede llevar una imagen incrustada (las tablas de pandas no la llevan y se conservan)
HTML_CON_IMAGEN = re.compile(r"<img|<svg|data:image|base64", re.IGNORECASE)


def _tiene_tag(cell: dict) -> bool:
    return TAG in cell.get("metadata", {}).get("tags", [])


def _texto(output: dict) -> str:
    partes = [output.get("text", "")] + [output.get("data", {}).get(k, "") for k in ("text/plain", "text/html")]
    partes += [output.get("evalue", "")] + list(output.get("traceback", []))
    return "".join("".join(p) if isinstance(p, list) else str(p) for p in partes)


def _html(output: dict) -> str:
    h = output.get("data", {}).get("text/html", "")
    return "".join(h) if isinstance(h, list) else str(h)


def _es_imagen(output: dict) -> bool:
    return any(k.startswith("image/") for k in output.get("data", {}))


def limpiar_notebook(notebook: dict) -> tuple[dict, list[str]]:
    """Limpia un notebook ya cargado (dict). Devuelve (notebook, descripción de lo que se quitó)."""
    quitado = []
    for index, cell in enumerate(notebook.get("cells", [])):
        outputs = cell.get("outputs") or []
        if not outputs:
            continue
        fuente = "".join(cell.get("source", []))
        tag = _tiene_tag(cell)
        con_caras = tag or bool(CODIGO_CON_CARAS.search(fuente))
        nuevas = []
        for out in outputs:
            if tag and HTML_CON_IMAGEN.search(_html(out)):
                quitado.append(f"celda {index}: HTML con imagen en una celda con tag '{TAG}'")
            elif con_caras and _es_imagen(out):
                quitado.append(f"celda {index}: imagen generada a partir de recortes/frames/Grad-CAM")
            elif TEXTO_PROHIBIDO.search(_texto(out)):
                quitado.append(f"celda {index}: log o ruta del script de descarga de FF++")
            else:
                nuevas.append(out)
        if len(nuevas) != len(outputs):
            cell["outputs"] = nuevas
    return notebook, quitado


def revisar(path: Path, limpiar: bool) -> list[str]:
    """Devuelve los problemas encontrados en un notebook; si `limpiar`, borra las salidas problemáticas."""
    texto = path.read_text(encoding="utf-8")
    notebook, quitado = limpiar_notebook(json.loads(texto))
    problemas = []
    if quitado and limpiar:
        path.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"limpio: {path} ({len(quitado)} salidas)")
    elif quitado:
        problemas += [f"{path}: {q}" for q in quitado]
    restante = json.dumps(notebook, ensure_ascii=False)
    if URL_PROHIBIDA in restante:
        problemas.append(f"{path}: contiene la URL del script de descarga de FF++ en el código (bórrala a mano)")
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
        print(f"OK: {len(notebooks)} notebooks sin salidas de caras ni log/URL del script de descarga.")
    return 1 if problemas else 0


if __name__ == "__main__":
    sys.exit(main())
