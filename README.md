# DeepLearning-G8-TA_Repository

Proyecto del Equipo 8: estudiar si Focal Loss mejora la robustez cross-manipulation de Xception para detectar deepfakes.

## Inicio rápido

El flujo experimental de referencia para Kaggle usa los notebooks de [`notebooks/kaggle/`](notebooks/README.md). Antes de ejecutar, revisa [la guía de ejecución](docs/GUIA_EJECUCION.md), el estado de tareas [E2](docs/E2_Tareas_programacion.md) y [E3](docs/E3_Tareas_programacion.md), y las decisiones vigentes en [DECISIONS](docs/DECISIONS.md).

Para preparar un entorno local con Python 3.10 o 3.11:

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# Linux/macOS: source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r environment/requirements.txt
```

La ejecución experimental completa requiere acceso autorizado a FF++ y un runtime compatible con PyTorch. No se suben al repositorio videos, frames, recortes, credenciales ni checkpoints grandes.

## Estructura

| Ruta | Propósito |
|---|---|
| `docs/` | Propuestas, contexto, decisiones, estructura y planes de entrega. |
| `configs/`, `environment/` | Configuración y dependencias. |
| `src/` | Lógica reutilizable, entrenamiento, evaluación y análisis. |
| `notebooks/kaggle/` | Orquestación de los experimentos en Kaggle. |
| `tests/`, `tools/` | Pruebas y utilidades de mantenimiento. |
| `results/` | Splits, métricas y resultados permitidos; los artefactos protegidos o pesados se excluyen. |

Consulta [docs/PROJECT_STRUCTURE.md](docs/PROJECT_STRUCTURE.md) para conocer los directorios con más detalle.

## Reproducibilidad y datos

Los notebooks llaman módulos de `src/` para evitar duplicar la lógica. PyTorch es el framework del proyecto: `timm` proporciona Xception y el código de pérdidas, entrenamiento y pruebas ya está implementado en ese ecosistema. La explicación de esta elección y de la separación entre notebooks y módulos está en [E3](docs/E3_Tareas_programacion.md).

FF++ se usa con acceso académico autorizado y no se redistribuye. Antes de versionar notebooks ejecutados, limpia y valida sus salidas:

```bash
python tools/limpiar_caras.py
python tools/limpiar_caras.py --check
```

La sincronización GitHub–Kaggle es manual: importa o clona el notebook en Kaggle y exporta los cambios al repositorio. Para rutas, comandos y opciones de los dos flujos disponibles, consulta [la guía de ejecución](docs/GUIA_EJECUCION.md).
