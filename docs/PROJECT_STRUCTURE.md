# Estructura del proyecto

El repositorio organiza por separado documentación, implementación, notebooks y resultados. La raíz se reserva para `README.md`, `.gitignore` y archivos de configuración de Git estrictamente necesarios.

## Directorios

| Ruta | Propósito |
|---|---|
| `docs/` | Propuesta histórica E1, contexto general, decisiones, guía de ejecución y seguimiento de entregas E2/E3. |
| `.github/workflows/` | Workflow `licencia-ffpp`: corre `tools/limpiar_caras.py --check` en cada push y falla si un notebook tiene caras o el log del script de descarga. |
| `configs/` | Configuración compartida de rutas, datos y entrenamiento (`base.yaml`). |
| `environment/` | Dependencias Python (`requirements.txt`). |
| `src/data_pipeline/` | Splits, extracción de frames, recortes MTCNN y Dataset para el pipeline CLI. |
| `src/kaggle_e2/` | Lógica modular importada por los notebooks de Kaggle (01–08): datos, preprocesamiento, SBI (`sbi.py`), entrenamiento, pérdidas, análisis, teardown y Grad-CAM. |
| `src/models/` | Constructor de Xception y funciones de pérdida reutilizables. |
| `src/teardown/` | Análisis de FLOPs/parámetros, campo receptivo, recursos y Grad-CAM del pipeline CLI. |
| `src/` | Puntos de entrada CLI de entrenamiento y evaluación; ver [guía de ejecución](GUIA_EJECUCION.md). |
| `notebooks/kaggle/` | Notebooks 01–08 que orquestan preparación, entrenamiento y análisis en Kaggle; índice en [`notebooks/README.md`](../notebooks/README.md). |
| `demo/` | Demo local en Gradio: detección de cara, probabilidad y Grad-CAM de los modelos de las Fases B y D ([`demo/README.md`](../demo/README.md)). Los pesos no se versionan. |
| `tests/` | Pruebas unitarias y controles de sanidad del código. |
| `tools/` | Generadores de notebooks y utilidades de limpieza de salidas protegidas. |
| `results/splits/` | Metadatos públicos de particiones e IDs fuente, sin medios del dataset. |
| `results/teardown/` | Tablas y figuras numéricas del análisis de arquitectura (CPU, GPU T4 y ERF sobre caras). |
| `results/metrics/` | Métricas e historiales livianos de las corridas (Fases B y C); la Fase D va en `results/metrics/fase_d/`, con su entorno. |
| `results/figures/` | Figuras permitidas; no guardar rostros derivados de FF++. |
| `results/checkpoints/` | Checkpoints locales (también los que usa la demo), excluidos de Git por ser derivados de FF++. |

## Principio de organización

Los notebooks son la entrada ejecutable y narrativa del flujo Kaggle. Importan de `src/` la lógica reutilizable para evitar duplicación y facilitar pruebas. El pipeline CLI permanece como ruta complementaria; los protocolos de ambas rutas están descritos en [DECISIONS](DECISIONS.md). Los comandos, rutas y pasos operativos están en la [guía de ejecución](GUIA_EJECUCION.md).

## Mantenimiento

- No añadir archivos sueltos al root salvo README, `.gitignore` y configuración de repositorio necesaria.
- Mantener los datos crudos/procesados, credenciales y checkpoints fuera de Git.
- Versionar solo metadatos, código y resultados permitidos por los términos del dataset.
- Actualizar los enlaces si cambia la ubicación de los documentos o scripts.
