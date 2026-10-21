# Notebooks

Los notebooks pensados para Kaggle viven en [`kaggle/`](kaggle/). El archivo exploratorio de FF++ es para inspección inicial con una muestra pequeña; no ejecuta el protocolo experimental completo definido en [E2](../docs/E2_Tareas_programacion.md).

Al crear notebooks de entrenamiento o análisis:

- importa funciones desde `src/` o mantén celdas breves que llamen a esos módulos;
- permite configurar rutas de Kaggle (`/kaggle/input/` para datos y `/kaggle/working/` para salidas);
- instala `environment/requirements.txt` (o selecciona las versiones compatibles con la imagen Kaggle activa) antes de importar módulos del proyecto;
- registra semilla, parámetros y hash del commit;
- nunca guardes credenciales ni copies el dataset al repositorio.

Kaggle mantiene versiones propias de los notebooks. Tras editar allí, descarga/exporta el `.ipynb` y actualiza esta carpeta con GitHub.
