# Cuaderno de decisiones

Registro de decisiones que afectan la reproducibilidad. Completar antes de las corridas oficiales; no inferir valores que aún no se han acordado.

| Tema | Decisión | Estado |
|---|---|---|
| Python / PyTorch / CUDA | Por definir según el entorno de ejecución | Pendiente |
| Detector facial | MTCNN de `facenet-pytorch`; seleccionar la cara de mayor área | Decisión inicial para T1; validar al procesar FF++ |
| Arquitectura y pesos de Xception | Confirmar nombre disponible en `timm` y origen de pesos | Pendiente |
| Normalización de entrada | E2 define estadísticas ImageNet `[0.485, 0.456, 0.406]` / `[0.229, 0.224, 0.225]`; la [implementación y ficha actual de `legacy_xception` en `timm`](https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/xception.py) registra `[0.5, 0.5, 0.5]` para ambos | Usar lo indicado por E2 por ahora; confirmar antes del entrenamiento con pesos preentrenados |
| Pérdida focal | `alpha=None` para que γ=0 coincida con BCE; revisar hipótesis de la propuesta | Propuesta inicial |
| Métrica de checkpoint | AUC de validación a nivel de video | Pendiente de confirmar |
| Extracción y recorte | 25 frames equiespaciados, recorte cuadrado con margen 1.3×, entrada 299×299; registrar y omitir frames sin cara | Decisión inicial para T1 |
| Splits oficiales | Pares `train.json`, `val.json`, `test.json` del repositorio FaceForensics de `ondyari`; se comprobará que los IDs de origen no crucen particiones | Fuente fijada en código; pendiente de materializar en Kaggle |
| Medición de FLOPs | Elegir y fijar `fvcore`, `ptflops` o `thop` | Pendiente |

## Registro de cambios

- 2026-10-21: se creó la estructura inicial del proyecto basada en E2.
- 2026-10-21: se organizaron documentos en `docs/`, el código en `src/` y dependencias en `environment/`; notebooks destinados a Kaggle en `notebooks/kaggle/`.
- 2026-10-21: se eligió MTCNN como detector inicial y se definió el formato CSV de manifiestos para el pipeline T1.
