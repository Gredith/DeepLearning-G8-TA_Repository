# Guía de ejecución

Esta guía reúne los comandos operativos para que el README principal quede breve. El flujo de referencia para E2/E3 son los notebooks Kaggle; el pipeline CLI es una ruta complementaria con algunos parámetros distintos. No mezcles sus resultados sin armonizar el protocolo. El estado de ejecución se registra en [E2](E2_Tareas_programacion.md), [E3](E3_Tareas_programacion.md) y [DECISIONS](DECISIONS.md).

## Notebooks de Kaggle (flujo de referencia)

Requisitos: acceso académico autorizado a FF++, un Dataset privado de Kaggle con los recursos aprobados, Internet para instalar dependencias y GPU compatible (T4 prevista). Los videos y recortes permanecen fuera del repositorio.

Orden sugerido:

1. `02_preprocesamiento_ffpp.ipynb`: prepara FF++ c23 según el protocolo registrado en `DECISIONS.md`; escribe los recortes de trabajo bajo `/kaggle/working/crops/`.
2. `03_entrenamiento_bce_vs_focal.ipynb`: entrena/compara BCE y Focal, y ejecuta evaluaciones y análisis implementados. Requiere los recortes preparados.
3. `01_teardown_xception.ipynb`: analiza arquitectura; sus secciones iniciales no requieren FF++, las posteriores sí usan caras y checkpoints.

Los notebooks clonan el repositorio en `/kaggle/working/repo`. Para que el notebook 03 encuentre los recortes, ejecuta 02 y 03 en la misma sesión o configura la ruta de los artefactos. Los valores del protocolo están en `DECISIONS.md`; no se asume que las celdas se hayan ejecutado con FF++ solo porque estén implementadas.

Antes de subir un notebook ejecutado:

```bash
python tools/limpiar_caras.py
python tools/limpiar_caras.py --check
```

Las celdas con rostros llevan el tag `caras`; la herramienta elimina salidas protegidas y `--check` detecta residuos o la URL prohibida del script oficial. Kaggle mantiene su propia versión del notebook. La sincronización con GitHub es manual: exporta los cambios y actualiza el `.ipynb` del repositorio.

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
