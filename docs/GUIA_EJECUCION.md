# Guía de ejecución

Esta guía reúne los comandos operativos para que el README principal quede breve. El flujo de referencia para E2/E3 son los notebooks Kaggle; el pipeline CLI es una ruta complementaria con algunos parámetros distintos. No mezcles sus resultados sin armonizar el protocolo. El estado de ejecución se registra en [E2](E2_Tareas_programacion.md), [E3](E3_Tareas_programacion.md) y [DECISIONS](DECISIONS.md).

## Notebooks de Kaggle (flujo de referencia)

Requisitos: acceso académico autorizado a FF++, un Dataset privado de Kaggle con los recursos aprobados, Internet para instalar dependencias y GPU compatible (T4 prevista). Los videos y recortes permanecen fuera del repositorio.

Orden (D-12):

1. **`02_preprocesamiento_ffpp.ipynb`** (una sola vez). Input: el Dataset privado con el script de descarga. GPU T4 e Internet activados. Ejecutar con **Save Version → Save & Run All (Commit)**, no en la sesión interactiva (~2–3 h). Su output son solo `crops.tar`, `manifest.csv` y `crops_meta.json`.
2. En esa versión: *Output* → **New Dataset** → `ffpp-crops-g8`, **Private**. Si la interfaz vuelve a fallar, activar `CREAR_POR_API = True` en la sección 5 del 02 (requiere el Secret `KAGGLE_KEY`).
3. **`03_entrenamiento_bce_vs_focal.ipynb`**. Input: `ffpp-crops-g8`. Extrae el tar a `/tmp`, valida los conteos, entrena BCE y Focal γ=2 (~20 min cada una en una T4) y deja en el output `resultados/` (JSON por corrida, CSV, figuras sin caras, `entorno.json`, `pip_freeze.txt`), `checkpoints/` y `privado/` (Grad-CAM).
4. **`01_teardown_xception.ipynb`**: las secciones 1–4 no usan datos. La sección 5 usa `ffpp-crops-g8` y, para Grad-CAM, el output del 03 como input.

Para llevar al repo los números de una corrida nueva, copiar `resultados/runs/*.json` y `comparacion_bce_vs_focal.csv` a `results/metrics/`.

### Licencia y sincronización Kaggle → GitHub

Cada *Save Version* sube el notebook a GitHub **con sus salidas**. Por eso:

- Ninguna celda muestra caras: las figuras con caras se guardan con `kaggle_io.save_private` en `privado/` (output privado de Kaggle). El log del script de descarga va a `privado/descarga.log`.
- Los notebooks de Kaggle y los Datasets (`ffpp-crops-g8`, el del script) deben ser **privados**.
- Segunda barrera: `tools/limpiar_caras.py` borra las salidas con el tag `caras`, las imágenes de celdas que leen recortes o Grad-CAM y cualquier log o ruta del script. El workflow `licencia-ffpp` lo corre con `--check` en cada push y falla si encuentra algo.

```bash
python tools/limpiar_caras.py
python tools/limpiar_caras.py --check
```

Después de cambiar un notebook en el repo, actualizarlo en Kaggle (*File → Import Notebook* o el enlace de GitHub) antes de la siguiente *Save Version*, para que la versión vieja no pise la nueva.

## Pipeline CLI complementario

Requiere FF++ autorizado, rutas configuradas en `configs/base.yaml` y dependencias de `environment/requirements.txt`. Desde la raíz del repositorio:

```bash
python src/data_pipeline/build_splits.py --config configs/base.yaml --download-official
python src/data_pipeline/extract_frames.py --config configs/base.yaml
python src/data_pipeline/crop_faces.py --config configs/base.yaml
python src/train.py --config configs/base.yaml --loss bce --run-name xception-bce
python src/train.py --config configs/base.yaml --loss focal --gamma 2 --run-name xception-focal-g2
python src/evaluate_cross.py --config configs/base.yaml --checkpoint BCE=results/checkpoints/xception-bce.pt --checkpoint Focal-g2=results/checkpoints/xception-focal-g2.pt
```

Para evaluar in-domain, usa `src/evaluate.py` con el checkpoint, split y manipulación deseados. El análisis de arquitectura se ejecuta con:

```bash
python src/teardown/flops_params.py --config configs/base.yaml
python src/teardown/receptive_field.py --config configs/base.yaml
python src/teardown/latency_memory.py --config configs/base.yaml --warmup 20 --iterations 100
python src/teardown/gradcam.py --config configs/base.yaml --bce-checkpoint results/checkpoints/xception-bce.pt --focal-checkpoint results/checkpoints/xception-focal-g2.pt
```

Para usar Kaggle con CLI, pasa `--data-root` para los datos, `--processed-root /kaggle/working/processed` para los recortes y `--results-root /kaggle/working/results` para salidas, además de `--crop-manifest` donde corresponda. En los comandos de splits, `--download-official` solo obtiene metadatos públicos de particiones; no descarga los videos FF++.

Los protocolos CLI y notebook difieren en número/formato de frames, entrenamiento y capa Grad-CAM, entre otros detalles. Consulta `DECISIONS.md` y `E2_Tareas_programacion.md` antes de interpretar o comparar sus salidas.

## Seguridad, licencia y artefactos

- No versionar videos, frames, recortes ni el script oficial de descarga de FF++.
- No guardar credenciales en notebooks o Git.
- Configuración, semilla, hash de commit y versiones deben acompañar cada corrida.
- Checkpoints grandes y figuras con caras no se suben a GitHub. Se versionan tablas y métricas permitidas.
