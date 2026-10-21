# Notebooks

Los notebooks pensados para Kaggle viven en [`kaggle/`](kaggle/):

| Notebook | Qué hace |
|---|---|
| `01_teardown_xception.ipynb` | Fase A: parámetros/FLOPs por flow, separable vs. estándar, RF/ERF, latencia y memoria; activaciones y Grad-CAM sobre caras (sección 5, Kaggle). |
| `02_preprocesamiento_ffpp.ipynb` | Descarga FF++ c23 con el script oficial (Dataset privado) y recorta caras con MTCNN. |
| `03_entrenamiento_bce_vs_focal.ipynb` | Fase B preliminar: BCE (γ=0) vs. Focal (γ=2), evaluación cross-manipulation y Grad-CAM. |
| `robustez-de-detectores-de-deepfakes-ta-dl.ipynb` | Explorador preliminar de FF++ con una muestra pequeña; no ejecuta el protocolo experimental. |

Los notebooks E2 se generan con `tools/build_nb01.py` y `tools/build_nb02_03.py` e importan los módulos de `src/kaggle_e2/`. Orden y detalles en el [README principal](../README.md#ejecución-e2-en-kaggle-notebooks).

**Licencia de FF++:** las celdas que muestran caras llevan el tag `caras`. Antes de versionar un notebook ejecutado, corre `python tools/limpiar_caras.py` desde la raíz y comprueba con `python tools/limpiar_caras.py --check`.

Al crear notebooks de entrenamiento o análisis:

- importa funciones desde `src/` o mantén celdas breves que llamen a esos módulos;
- permite configurar rutas de Kaggle (`/kaggle/input/` para datos y `/kaggle/working/` para salidas);
- instala `environment/requirements.txt` (o selecciona las versiones compatibles con la imagen Kaggle activa) antes de importar módulos del proyecto;
- registra semilla, parámetros y hash del commit;
- nunca guardes credenciales ni copies el dataset al repositorio.

Kaggle mantiene versiones propias de los notebooks. Tras editar allí, descarga/exporta el `.ipynb` y actualiza esta carpeta con GitHub.
