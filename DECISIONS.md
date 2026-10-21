# Cuaderno de decisiones

Registro de decisiones que afectan la reproducibilidad. Completar antes de las corridas oficiales; no inferir valores que aún no se han acordado.

| Tema | Decisión | Estado |
|---|---|---|
| Python / PyTorch / CUDA | Por definir según el entorno de ejecución | Pendiente |
| Detector facial | MTCNN o RetinaFace | Pendiente |
| Arquitectura y pesos de Xception | Confirmar nombre disponible en `timm` y origen de pesos | Pendiente |
| Pérdida focal | `alpha=None` para que γ=0 coincida con BCE; revisar hipótesis de la propuesta | Propuesta inicial |
| Métrica de checkpoint | AUC de validación a nivel de video | Pendiente de confirmar |
| Extracción y recorte | Frames, margen y política para caras no detectadas | Pendiente |
| Medición de FLOPs | Elegir y fijar `fvcore`, `ptflops` o `thop` | Pendiente |

## Registro de cambios

- 2026-10-21: se creó la estructura inicial del proyecto basada en E2.
