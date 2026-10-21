# DeepLearning-G8-TA_Repository

Proyecto del Equipo 8 para estudiar la robustez cross-manipulation de Xception con focal loss en detección de deepfakes.

## Estado

El repositorio contiene la estructura inicial y los puntos de entrada del pipeline. Las tareas experimentales y su estado se mantienen en [E2_Tareas_programacion.md](E2_Tareas_programacion.md). Antes de usar pesos preentrenados o datos de FaceForensics++, revisa [DECISIONS.md](DECISIONS.md) y las condiciones de acceso/licencia del conjunto de datos.

## Requisitos

Se recomienda Python 3.10 o 3.11 y una instalación de PyTorch compatible con el equipo (CPU o CUDA). Instala las dependencias del proyecto con:

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# Linux/macOS: source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Las versiones deberán fijarse tras confirmar el entorno de ejecución y antes de producir resultados comparables. No se incluyen datos ni checkpoints en Git.

## Preparación de datos

1. Obtén FF++ siguiendo el procedimiento oficial y acepta sus términos de uso. No copies videos ni frames al repositorio.
2. Edita `configs/base.yaml` para indicar la ubicación local de los datos y los directorios de salida.
3. Implementa y ejecuta las etapas del pipeline en orden: `data_pipeline/build_splits.py`, `extract_frames.py` y `crop_faces.py`.
4. Conserva los splits por ID de video y verifica que no exista solapamiento entre train, validación y test.

Los scripts de datos son actualmente puntos de entrada de la estructura; las descargas, la extracción y los recortes aún están pendientes.

## Entrenamiento y evaluación

Una vez completado el pipeline, el flujo previsto es:

```bash
python train.py --config configs/base.yaml --loss bce --run-name xception-bce
python train.py --config configs/base.yaml --loss focal --gamma 2 --run-name xception-focal-g2
python evaluate.py --config configs/base.yaml --checkpoint results/checkpoints/xception-bce.pt
```

Estos puntos de entrada todavía no ejecutan entrenamiento ni evaluación completos. Los artefactos se escribirán bajo `results/`: checkpoints, métricas, figuras y splits. Sigue E2 para la implementación y aceptación de cada etapa.

## Reproducibilidad

- Registra decisiones de datos/modelo y versiones en `DECISIONS.md` y `requirements.txt`.
- Mantén iguales datos, splits, semilla y configuración entre BCE y focal; cambia únicamente la pérdida.
- Guarda la configuración y el hash de Git junto con cada corrida.
- Consulta los comandos y criterios de aceptación detallados en [E2_Tareas_programacion.md](E2_Tareas_programacion.md).
