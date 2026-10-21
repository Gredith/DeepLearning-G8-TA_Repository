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
| `src/` | Puntos de entrada para entrenamiento y evaluación. |
| `tests/` | Pruebas de aceptación del código (pendientes de completar). |
| `notebooks/kaggle/` | Notebooks diseñados para explorar o ejecutar etapas en Kaggle. El explorador de FF++ es una herramienta preliminar, no el pipeline oficial de experimentos. |
| `results/` | Salidas reproducibles: checkpoints, métricas, figuras y splits; los datos y artefactos pesados se excluyen de Git. |

Para el detalle de cada módulo y el flujo recomendado, consulta [docs/PROJECT_STRUCTURE.md](docs/PROJECT_STRUCTURE.md). El avance y los pendientes se mantienen en [E2](docs/E2_Tareas_programacion.md); las decisiones técnicas están en [DECISIONS](docs/DECISIONS.md).

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
3. Completa T1 en E2: splits por ID de video, extracción, recortes y `Dataset`.
4. Ejecuta los puntos de entrada desde la raíz del repositorio:

```bash
python src/train.py --config configs/base.yaml --loss bce --run-name xception-bce
python src/train.py --config configs/base.yaml --loss focal --gamma 2 --run-name xception-focal-g2
python src/evaluate.py --config configs/base.yaml --checkpoint results/checkpoints/xception-bce.pt
```

Los comandos son la interfaz prevista; entrenamiento, evaluación y pipeline siguen pendientes de implementación. Consulta E2 antes de interpretar estas salidas.

## Uso con Kaggle

Los notebooks de `notebooks/kaggle/` se versionan en GitHub y se pueden importar o clonar en Kaggle. Kaggle ejecuta su propia copia/versionado del notebook: para devolver cambios al repositorio, guarda/exporta el `.ipynb` y súbelo mediante Git. Los datos privados deben adjuntarse como Kaggle Dataset o almacenarse fuera del repositorio; usa `/kaggle/input/` para leerlos y `/kaggle/working/` para salidas temporales. La sincronización automática GitHub–Kaggle aún no está configurada.

El notebook exploratorio actual descarga una muestra pequeña para inspección visual. No reemplaza los splits oficiales, el preprocesamiento completo ni las corridas registradas requeridas por E2.

## Reproducibilidad

- Registra detector, arquitectura, versiones y demás decisiones en `docs/DECISIONS.md`.
- Mantén iguales datos, splits, semilla y configuración entre BCE y focal; cambia solo la pérdida.
- Guarda la configuración y el hash del commit junto con cada corrida.
- Usa [E2](docs/E2_Tareas_programacion.md) como lista de tareas pendientes y criterios de aceptación.
