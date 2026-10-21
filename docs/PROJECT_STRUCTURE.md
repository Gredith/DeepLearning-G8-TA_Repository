# Estructura del proyecto

El repositorio separa documentación, código, configuración, notebooks y resultados para que el trabajo local y las ejecuciones en Kaggle compartan una organización predecible. En el root solo deben permanecer `README.md` y `.gitignore`.

## Directorios

| Directorio | Propósito y contenido esperado |
|---|---|
| `docs/` | Documentación humana: propuesta E1, tareas y criterios de E2, decisiones técnicas y esta guía. |
| `environment/` | Dependencias reproducibles, incluyendo `requirements.txt`; fijar versiones antes de las corridas oficiales. |
| `configs/` | Archivos YAML compartidos. `base.yaml` centraliza semillas, parámetros de entrenamiento y rutas relativas/configurables. |
| `src/data_pipeline/` | Descarga y auditoría de splits oficiales de video, manifiesto de clips, extracción equiespaciada, recortes MTCNN precalculados y `Dataset` con normalización ImageNet. |
| `src/models/` | Constructor de Xception y pérdidas (BCE/focal). |
| `src/teardown/` | Scripts de análisis de arquitectura, FLOPs, campo receptivo, rendimiento y explicabilidad. |
| `src/train.py`, `src/evaluate.py`, `src/evaluate_cross.py` | Interfaces de línea de comandos para entrenar, medir in-domain y generar la tabla cross-manipulation. Se invocan desde la raíz. |
| `tests/` | Pruebas unitarias y de sanidad: pérdidas, formas, métricas y ausencia de fuga entre splits. |
| `notebooks/kaggle/` | Notebooks autocontenidos para ingestión/exploración o experimentos que requieran el entorno Kaggle. Deben leer datos de `/kaggle/input/` y escribir salidas en `/kaggle/working/`. |
| `results/splits/` | Manifiestos/listas de IDs por partición, sin datos personales o medios del dataset. |
| `results/metrics/` | Métricas JSON/CSV y tablas de análisis. |
| `results/figures/` | Figuras producidas por los análisis y notebooks. |
| `results/checkpoints/` | Pesos de modelos. Pesados y excluidos de Git por defecto. |

## Política para notebooks y Kaggle

- Mantener notebooks versionados bajo `notebooks/kaggle/`, no en la raíz.
- Un notebook de exploración sirve para inspeccionar datos y documentar acceso; no debe convertirse en la fuente de verdad de la lógica de entrenamiento.
- La lógica que se reutilice entre notebook, entrenamiento y evaluación debe migrar a `src/`.
- La fuente pública de los JSON de split es [`ondyari/FaceForensics/dataset/splits`](https://github.com/ondyari/FaceForensics/tree/master/dataset/splits); el script descarga solo esos metadatos y guarda sus hashes. FF++ requiere aprobación de acceso y no se descarga desde el pipeline.
- Registrar parámetros de corrida y hash del commit. Mantener las credenciales fuera del notebook y del repositorio.
- GitHub y Kaggle son copias/versiones distintas salvo que se configure una sincronización explícita. Importar/actualizar el notebook en Kaggle y exportar/commit de vuelta es el flujo manual actual.
- No almacenar FF++ en GitHub. En Kaggle usar un Dataset privado autorizado y tratar `/kaggle/input/` como solo lectura.

## Reglas de mantenimiento

1. No añadir archivos sueltos al root salvo README, `.gitignore` y archivos de configuración de repositorio estrictamente necesarios.
2. Guardar datos crudos/procesados fuera de Git; los directorios locales `data/` y `datasets/` están ignorados.
3. Mantener manifiestos y tablas pequeñas necesarios para E2; excluir checkpoints y figuras con imágenes derivadas del dataset.
4. Actualizar enlaces y comandos en README si cambia la ubicación de un archivo.
