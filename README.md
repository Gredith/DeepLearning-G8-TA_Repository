# DeepLearning-G8-TA_Repository

Proyecto del Equipo 8 para estudiar la robustez cross-manipulation de Xception con focal loss en detección de deepfakes.

## Estructura del repositorio

El directorio raíz se reserva para este README y `.gitignore`. La documentación, el código y los artefactos tienen directorios propios:

| Ruta | Contenido |
|---|---|
| `docs/` | Propuesta E1, plan vivo de tareas E2 y cuaderno de decisiones experimentales. |
| `environment/` | Dependencias Python del proyecto. |
| `configs/` | Configuración base de rutas, datos, modelo, entrenamiento y pérdida. |
| `src/data_pipeline/` | Preparación de splits por video, extracción, recorte y carga de datos. |
| `src/models/` | Constructor de Xception y funciones de pérdida. |
| `src/teardown/` | Análisis de FLOPs, campo receptivo, latencia, memoria y Grad-CAM. |
| `src/` | Puntos de entrada para entrenamiento, evaluación in-domain y comparación cross-manipulation. |
| `src/kaggle_e2/` | Módulos que usan los notebooks E2: datos, recorte, pérdida focal, entrenamiento, teardown y Grad-CAM. |
| `tests/` | Pruebas del pipeline y criterios BCE/focal; las que requieren PyTorch se activan en el entorno Kaggle. |
| `notebooks/kaggle/` | Notebooks E2 (`01`–`03`: teardown, preprocesamiento, BCE vs. focal) y el explorador preliminar de FF++. |
| `tools/` | Generadores de los notebooks E2 y `limpiar_caras.py` (licencia de FF++). |
| `results/` | Splits, tablas del teardown (`results/teardown/`) y métricas por corrida (`results/runs/`); checkpoints y figuras con caras del dataset se excluyen de Git. |

Para el detalle de cada módulo y el flujo recomendado, consulta [docs/PROJECT_STRUCTURE.md](docs/PROJECT_STRUCTURE.md). El avance y los pendientes se mantienen en [E2](docs/E2_Tareas_programacion.md); las decisiones técnicas, hipótesis y uso de IA están en [DECISIONS](docs/DECISIONS.md).

## Ejecución E2 en Kaggle (notebooks)

Los resultados del avance E2 se producen con los notebooks de `notebooks/kaggle/`. Cada uno clona este repositorio en `/kaggle/working/repo`; en Kaggle activa **GPU T4** e **Internet**.

1. **Script de descarga de FF++:** súbelo como **Kaggle Dataset privado** (*Add Input → Upload*). Es el que llega con el correo de aprobación de acceso; no se versiona ni se enlaza aquí (licencia).
2. **[`02_preprocesamiento_ffpp.ipynb`](notebooks/kaggle/02_preprocesamiento_ffpp.ipynb):** descarga FF++ c23 y genera `/kaggle/working/crops/` (MTCNN, 16 frames por video) para 100/30/70 pares oficiales (D-05).
3. **[`03_entrenamiento_bce_vs_focal.ipynb`](notebooks/kaggle/03_entrenamiento_bce_vs_focal.ipynb):** entrena BCE (γ=0) y Focal (γ=2) con el mismo protocolo y evalúa in-domain y cross-manipulation. Escribe `results/runs/*.json`, `results/comparacion_bce_vs_focal.csv` y las figuras.
4. **[`01_teardown_xception.ipynb`](notebooks/kaggle/01_teardown_xception.ipynb):** Fase A. Las secciones 1–4 no necesitan datos; la sección 5 usa los recortes y los checkpoints.

Si 02 y 03 se ejecutan en la misma sesión, 03 encuentra los recortes en `/kaggle/working/crops`. Antes de subir a GitHub un notebook ejecutado, corre `python tools/limpiar_caras.py` (ver Licencias).

### Pipeline CLI completo

Además de los notebooks E2, el repositorio conserva el pipeline modular de `src/` para preparar el conjunto completo y reproducir experimentos desde la línea de comandos. Requiere FF++ c23 obtenido con acceso autorizado y ubicado según `configs/base.yaml`:

```bash
python src/data_pipeline/build_splits.py --config configs/base.yaml --download-official
python src/data_pipeline/extract_frames.py --config configs/base.yaml
python src/data_pipeline/crop_faces.py --config configs/base.yaml
python src/train.py --config configs/base.yaml --loss bce --run-name xception-bce
python src/train.py --config configs/base.yaml --loss focal --gamma 2 --run-name xception-focal-g2
python src/evaluate_cross.py --config configs/base.yaml --checkpoint BCE=results/checkpoints/xception-bce.pt --checkpoint Focal-g2=results/checkpoints/xception-focal-g2.pt
```

Los análisis de FLOPs/parámetros, campo receptivo y latencia/memoria se ejecutan con `src/teardown/flops_params.py`, `src/teardown/receptive_field.py` y `src/teardown/latency_memory.py`. Los dos flujos comparten el modelo Xception y los splits oficiales; difieren en el subconjunto de datos y protocolo experimental. Los resultados publicados para este E2 corresponden a los notebooks 01–03.

## Preparar el entorno

Se recomienda Python 3.10 o 3.11 y una instalación de PyTorch compatible con el equipo (CPU o CUDA). Desde la raíz del repositorio:

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# Linux/macOS: source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r environment/requirements.txt
```

Las versiones de dependencias deben fijarse después de validar el entorno y antes de producir resultados comparables. No se incluyen datos ni checkpoints en Git.

## Datos y ejecución

1. Obtén FF++ por el procedimiento oficial y acepta sus términos. No subas videos ni frames al repositorio.
2. Configura rutas locales en `configs/base.yaml`.
3. Genera/valida los splits oficiales por pares de IDs y el inventario de videos:

```bash
python src/data_pipeline/build_splits.py --config configs/base.yaml --download-official
python src/data_pipeline/extract_frames.py --config configs/base.yaml
python src/data_pipeline/crop_faces.py --config configs/base.yaml
```

La descarga de splits es solo de metadatos públicos; no descarga FF++. Adjunta/descarga el dataset aprobado aparte. El layout esperado es `original_sequences/youtube/c23/videos/` y `manipulated_sequences/<manipulacion>/c23/videos/` bajo `data_root`.

4. Ejecuta los puntos de entrada desde la raíz del repositorio:

```bash
python src/train.py --config configs/base.yaml --loss bce --run-name xception-bce
python src/train.py --config configs/base.yaml --loss focal --gamma 0 --run-name xception-focal-g0
python src/train.py --config configs/base.yaml --loss focal --gamma 2 --run-name xception-focal-g2
python src/evaluate.py --config configs/base.yaml --checkpoint results/checkpoints/xception-bce.pt --split test --manipulation Deepfakes
```

Para la evaluación cross-manipulation, repite la última orden con `--manipulation Face2Face`, `FaceSwap` o `NeuralTextures`; cada corrida incluye las caras `original` del mismo split.

Una vez existan ambos checkpoints, genera las tablas cross-manipulation y los mapas Grad-CAM:

```bash
python src/evaluate_cross.py --config configs/base.yaml --checkpoint BCE=results/checkpoints/xception-bce.pt --checkpoint Focal-g2=results/checkpoints/xception-focal-g2.pt
python src/teardown/gradcam.py --config configs/base.yaml --bce-checkpoint results/checkpoints/xception-bce.pt --focal-checkpoint results/checkpoints/xception-focal-g2.pt
```

Los comandos de análisis de arquitectura escriben sus tablas y figuras en `results/metrics/` y `results/figures/`:

```bash
python src/teardown/flops_params.py --config configs/base.yaml
python src/teardown/receptive_field.py --config configs/base.yaml
python src/teardown/latency_memory.py --config configs/base.yaml --warmup 20 --iterations 100
```

Para ejecutar T8, lanza otra corrida con `--loss focal --gamma 1` y usa ese checkpoint en `evaluate_cross.py`. La ejecución en c40/DFDC (T9) requiere agregar esos datos y no forma parte del pipeline base de c23.

En Kaggle, usa las opciones de ruta para leer el Dataset adjunto desde `/kaggle/input/` y escribir el preprocesamiento/resultados en `/kaggle/working/`:

```bash
python src/data_pipeline/build_splits.py --config configs/base.yaml --download-official --data-root /kaggle/input/ff-plus-plus-c23/FF_data
python src/data_pipeline/extract_frames.py --config configs/base.yaml --data-root /kaggle/input/ff-plus-plus-c23/FF_data --processed-root /kaggle/working/processed
python src/data_pipeline/crop_faces.py --config configs/base.yaml --processed-root /kaggle/working/processed
python src/train.py --config configs/base.yaml --loss bce --run-name xception-bce --processed-root /kaggle/working/processed --results-root /kaggle/working/results
```

Para las siguientes corridas/evaluaciones en Kaggle, pasa también `--processed-root /kaggle/working/processed`, `--crop-manifest /kaggle/working/processed/crop_manifest.csv` y `--results-root /kaggle/working/results` a cada comando que los admita.

Los puntos de entrada de datos, entrenamiento y evaluación ya están implementados; la ejecución integral y los resultados quedan pendientes de un runtime compatible y acceso autorizado a FF++. Consulta E2 antes de interpretar salidas experimentales.

## Uso con Kaggle

Los notebooks de `notebooks/kaggle/` se versionan en GitHub y se pueden importar o clonar en Kaggle. Kaggle ejecuta su propia copia/versionado del notebook: para devolver cambios al repositorio, guarda/exporta el `.ipynb` y súbelo mediante Git. Los datos privados deben adjuntarse como Kaggle Dataset o almacenarse fuera del repositorio; usa `/kaggle/input/` para leerlos y `/kaggle/working/` para salidas temporales. La sincronización automática GitHub–Kaggle aún no está configurada.

El notebook exploratorio actual descarga una muestra pequeña para inspección visual. No reemplaza los splits oficiales, el preprocesamiento completo ni las corridas registradas requeridas por E2.

## Licencias y datos

- **FaceForensics++** (Rössler et al., 2019): uso académico bajo los términos de TU München. Videos, frames, recortes de caras y el script oficial de descarga **no se redistribuyen**: no se suben a este repositorio y el script no se enlaza.
- Las celdas de los notebooks que muestran caras llevan el tag `caras`. `python tools/limpiar_caras.py` borra sus salidas y `python tools/limpiar_caras.py --check` falla si queda alguna o si un notebook contiene la URL del script. Las figuras con caras están en `.gitignore`.
- **Splits:** `results/splits/{train,val,test}.json` son los metadatos públicos de [`ondyari/FaceForensics`](https://github.com/ondyari/FaceForensics/tree/master/dataset/splits).
- **DFDC Preview** (Dolhansky et al., 2019): términos de uso de Meta.
- **Pesos ImageNet de Xception:** `timm` (Apache-2.0).
- **Uso responsable:** los videos contienen rostros de personas reales. Se usan solo para investigación, sin intentar identificar a las personas y sin generar deepfakes nuevos.

## Referencias principales

- A. Rössler et al., "FaceForensics++: Learning to Detect Manipulated Facial Images," ICCV 2019.
- F. Chollet, "Xception: Deep Learning with Depthwise Separable Convolutions," CVPR 2017.
- T.-Y. Lin et al., "Focal Loss for Dense Object Detection," ICCV 2017.
- W. Luo et al., "Understanding the Effective Receptive Field in Deep Convolutional Neural Networks," NeurIPS 2016.
- R. R. Selvaraju et al., "Grad-CAM," ICCV 2017.

## Reproducibilidad

- Registra detector, arquitectura, versiones y demás decisiones en `docs/DECISIONS.md`.
- Mantén iguales datos, splits, semilla y configuración entre BCE y focal; cambia solo la pérdida.
- Guarda la configuración y el hash del commit junto con cada corrida.
- Usa [E2](docs/E2_Tareas_programacion.md) como lista de tareas pendientes y criterios de aceptación.
