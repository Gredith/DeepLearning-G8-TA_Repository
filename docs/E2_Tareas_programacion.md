# E2 · Registro de tareas de programación

**Proyecto:** Robustez cross-manipulation de Xception con Focal Loss (Equipo 8)
**Entrega E2:** miércoles 21 de octubre de 2026
**Uso:** registro del desarrollo y de la evidencia del E2. Las próximas tareas de entrega se gestionan en [E3](E3_Tareas_programacion.md).

> **Cómo leer el estado:** «Implementado» indica que hay código/notebook; «Ejecutado» indica evidencia producida en el entorno y con los datos del experimento. Una prueba sintética valida el flujo de software, pero no cuenta como resultado experimental con FF++.

## Resumen de notebooks

| Notebook | Alcance | Estado de evidencia |
|---|---|---|
| [`01_teardown_xception.ipynb`](../notebooks/kaggle/01_teardown_xception.ipynb) | Teardown de Xception, FLOPs/parámetros, campo receptivo, latencia/memoria y secciones dependientes de datos para activaciones/ERF/Grad-CAM. | Tablas/figuras de CPU están en `results/teardown/`. Medidas GPU y análisis sobre caras quedan pendientes de ejecución/revisión en Kaggle. |
| [`02_preprocesamiento_ffpp.ipynb`](../notebooks/kaggle/02_preprocesamiento_ffpp.ipynb) | Splits por video, extracción de 16 frames, MTCNN y recortes FF++ c23. | Notebook implementado; falta ejecutar con Dataset privado autorizado, revisar conteos/recortes y guardar manifiestos permitidos. |
| [`03_entrenamiento_bce_vs_focal.ipynb`](../notebooks/kaggle/03_entrenamiento_bce_vs_focal.ipynb) | Entrenamiento BCE/Focal, métricas in-domain/cross-manipulation y Grad-CAM mediante `src/kaggle_e2/`. | Código implementado; S-01 prueba el flujo con datos sintéticos. Faltan corridas entrenadas con FF++ y sus resultados. |

La ruta de referencia para las salidas E2 son los notebooks y `src/kaggle_e2/`. El pipeline CLI en `src/` es complementario y tiene diferencias de protocolo. Para detalles de esas diferencias, decisiones y configuraciones vigentes, consulta [DECISIONS](DECISIONS.md) y la [guía de ejecución](GUIA_EJECUCION.md).

## Estado por tarea

| Tarea | Código | Ejecución/evidencia pendiente |
|---|---|---|
| **T1 · Datos**: splits oficiales por video, extracción, MTCNN, dataset, conteos y controles de fuga. | Implementado en notebook y módulos; JSON/hash de splits versionados. | Correr con FF++ autorizado; revisar grillas, conteos, tasa de caras no detectadas y manifiestos. |
| **T2 · Xception/BCE**: entrenamiento reproducible, evaluación por frame/video, checkpoint e historial. | Implementado. | Correr baseline real y guardar métricas/checkpoint de forma permitida. |
| **T3 · Focal Loss**: equivalencia γ=0 con BCE, casos fáciles/difíciles, estabilidad y gradientes. | Implementado y cubierto por pruebas; S-01 sintético de extremo a extremo. | Ejecutar pruebas en runtime con PyTorch y entrenar BCE (γ=0) y Focal (γ=2) con FF++. |
| **T4 · Teardown**: parámetros/FLOPs por bloque y comparación separable/estándar. | Implementado; resultados CPU guardados en `results/teardown/`. | Validar/añadir corrida en GPU según alcance E3. |
| **T5 · Grad-CAM**: selección emparejada, visualización e IDs reproducibles. | Implementado; CLI usa `conv4`, notebooks E2 usan `block12` (D-10). | Generar/revisar mapas usando checkpoints entrenados y datos autorizados. |
| **T6 · Cross-manipulation**: evaluación en Face2Face, FaceSwap y NeuralTextures, AUC por frame/video y caída relativa. | Implementado. | Correr con checkpoints reales, llenar tabla y discutir el resultado. |
| **T7 · Campo receptivo/recursos**: RF, latencia y memoria. | Implementado; hay tablas CPU disponibles. | Ejecutar mediciones GPU y completar interpretación dependiente de datos. |
| **T8 · γ=1** | Opcional, no ejecutado. | Prioridad posterior a la comparación principal; ver E3. |
| **T9 · c40/DFDC** | Opcional, no ejecutado. | Solo si acceso y presupuesto lo permiten; ver E3. |

## Documento de evidencias E2

Además del código/notebooks, E2 solicita un documento Word o Google Docs que describa el avance y enlace el repositorio. El documento debe resumir el procedimiento efectivamente realizado en cada notebook y acompañarlo con evidencia disponible; no presentar como ejecutada una celda que solo está implementada.

- [ ] **Preparar y entregar el documento de evidencias E2** en Word o Google Docs usando la [estructura propuesta](E2_Evidencias_estructura.md); registrar aquí su enlace o ubicación.
- [ ] Incluir el enlace al GitHub y la versión/commit revisado.
- [ ] Resumir el propósito y procedimiento de los notebooks 01–03, las entradas/configuración y las salidas producidas.
- [ ] Incluir tablas, capturas o métricas pertinentes, distinguiendo teardown CPU, prueba sintética y resultados experimentales con FF++.
- [ ] Declarar ejecuciones pendientes, limitaciones y datos no disponibles.

El documento Word/Google Docs es el reporte de avance; este Markdown mantiene su checklist y referencia. La [estructura propuesta](E2_Evidencias_estructura.md) sirve como plantilla, la guía de ejecución explica los comandos y el reporte presenta qué se hizo y qué evidencia se obtuvo.

## Criterio de aceptación E2

- [ ] Pipeline de datos corrido con FF++ y controles visuales/conteos revisados.
- [ ] Baseline BCE y Focal γ=2 entrenados con protocolo comparable y métricas in-domain guardadas.
- [ ] Evaluación cross-manipulation con tabla de resultados y análisis de la hipótesis.
- [x] Implementación de teardown CPU y código para pérdidas, Grad-CAM, evaluación y mediciones de arquitectura.
- [ ] Secciones que requieren GPU/datos ejecutadas y evidencia versionada sin materiales protegidos.
- [ ] Documento de evidencias E2 entregado (o entrega confirmada) con procedimiento de notebooks y enlace al repositorio.
- [ ] Versiones del entorno registradas; notebooks que se presenten como reproducibles ejecutados de arriba abajo.

El seguimiento de continuación, la demo y el informe final están en [E3](E3_Tareas_programacion.md). No se deben completar casillas experimentales solo porque exista su implementación.
