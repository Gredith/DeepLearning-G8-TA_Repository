# Notebooks de Kaggle

Los notebooks ejecutables están en [`kaggle/`](kaggle/):

| Notebook | Función |
|---|---|
| `01_teardown_xception.ipynb` | Fase A: parámetros/FLOPs, campo receptivo y recursos; incluye secciones que requieren datos para activaciones/ERF/Grad-CAM. |
| `02_preprocesamiento_ffpp.ipynb` | Preparación de FF++ c23 autorizado y recortes faciales. |
| `03_entrenamiento_bce_vs_focal.ipynb` | Comparación BCE/Focal y evaluación in-domain/cross-manipulation. |
| `robustez-de-detectores-de-deepfakes-ta-dl.ipynb` | Exploración preliminar; no sustituye el protocolo experimental. |

Los notebooks `01`–`03` importan funciones de `src/kaggle_e2/`. Para orden, rutas de datos, dependencias y comandos, consulta la [guía de ejecución](../docs/GUIA_EJECUCION.md). El estado de evidencia está en [E2](../docs/E2_Tareas_programacion.md) y las decisiones experimentales en [DECISIONS](../docs/DECISIONS.md).

Antes de versionar un notebook ejecutado, elimina las salidas con rostros y valida el archivo:

```bash
python tools/limpiar_caras.py
python tools/limpiar_caras.py --check
```

Los datos privados y credenciales no se guardan en el notebook ni en Git. La sincronización entre Kaggle y GitHub es manual: exporta los cambios de Kaggle y actualiza el `.ipynb` del repositorio.
