# Notebooks de Kaggle

Los notebooks ejecutables están en [`kaggle/`](kaggle/):

| Notebook | Función |
|---|---|
| `01_teardown_xception.ipynb` | Fase A: parámetros/FLOPs, campo receptivo y recursos; incluye secciones que requieren datos para activaciones/ERF/Grad-CAM. |
| `02_preprocesamiento_ffpp.ipynb` | Descarga FF++ c23 autorizado y recorta caras; su output (`crops.tar` + `manifest.csv`) se convierte en el Dataset privado `ffpp-crops-g8`. |
| `03_entrenamiento_bce_vs_focal.ipynb` | Usa `ffpp-crops-g8` como input: comparación BCE/Focal y evaluación in-domain/cross-manipulation. |
| `04_barrido_gamma_semillas.ipynb` | Barrido γ ∈ {0, 1, 2, 5} × semillas {42, 43, 44} con `ffpp-crops-g8`; reanudable usando su output anterior como input. |
| `05_analisis_final.ipynb` | Sin GPU ni datos: tabla final (media ± desv.), IC 95 % por bootstrap de Δ AUC y figuras del informe, desde los JSON y predicciones del 03/04. |
| `06_diagnostico_gradcam.ipynb` | Grad-CAM BCE vs. Focal (selección D-10) con figuras solo en `privado/` y métricas numéricas por región para explicar FaceSwap < 0.5. |
| `robustez-de-detectores-de-deepfakes-ta-dl.ipynb` | Exploración preliminar; no sustituye el protocolo experimental. |

Los notebooks `01`–`03` importan funciones de `src/kaggle_e2/`. Para orden, rutas de datos, dependencias y comandos, consulta la [guía de ejecución](../docs/GUIA_EJECUCION.md). El estado de evidencia está en [E2](../docs/E2_Tareas_programacion.md) y las decisiones experimentales en [DECISIONS](../docs/DECISIONS.md).

Ninguna celda muestra caras de FF++: las figuras con caras se guardan en `privado/` dentro del output de Kaggle (D-12). Igual, antes de versionar un notebook ejecutado, valida el archivo (el workflow `licencia-ffpp` también lo hace en cada push):

```bash
python tools/limpiar_caras.py
python tools/limpiar_caras.py --check
```

Los datos privados y credenciales no se guardan en el notebook ni en Git. Kaggle sube cada *Save Version* a GitHub con sus salidas; por eso los notebooks están hechos para que esas salidas no tengan caras ni el log del script de descarga.
