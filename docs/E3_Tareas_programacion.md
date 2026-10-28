# E3 · Tareas de programación y entrega

**Proyecto:** Robustez cross-manipulation de Xception con Focal Loss (Equipo 8)
**Entrega:** miércoles 11 de noviembre de 2026
**Objetivo:** entregar una demo reproducible y el informe final de 4–8 páginas con la plantilla WAIMLAp.
**Plan de trabajo:** fases A (teardown), B (intervención BCE/Focal) y C (evaluación bajo cambios de manipulación/dataset), definidas en [CONTEXTO_PROYECTO.md](CONTEXTO_PROYECTO.md).

> **Estado al 27/10/2026:** el código del flujo E2 está implementado en los notebooks `01`–`03` y `src/kaggle_e2/`. Hay resultados de teardown en CPU y una prueba sintética de extremo a extremo (S-01), pero la documentación y los artefactos revisados no muestran corridas de entrenamiento con FF++ ni métricas experimentales BCE/Focal. Por tanto, la prioridad inmediata es ejecutar y verificar el flujo principal con el acceso autorizado antes de dedicar tiempo a extensiones. Este plan queda como registro vivo hasta el E3; el histórico de E2 está en [E2_Tareas_programacion.md](E2_Tareas_programacion.md).

## Estado actual y siguiente acción

| Frente | Estado al 27/10 | Pendiente/decisión |
|---|---|---|
| Implementación | Notebooks 01–03 y módulos `src/kaggle_e2/` presentes; scripts auxiliares y pipeline CLI también en el repositorio. | Usar los notebooks como ruta de referencia del entregable y mantener documentadas las diferencias del CLI. |
| Fase A · teardown | Tablas/figuras CPU guardadas en `results/teardown/`. | Correr las mediciones que requieren GPU/caras y revisar su interpretación. |
| Fase B · entrenamiento | Hay prueba sintética S-01 registrada; no acredita resultados sobre FF++. | **Siguiente paso:** ejecutar notebook 02 con el Dataset privado autorizado, revisar conteos y recortes, y continuar con notebook 03 para BCE/γ=0 y Focal/γ=2. |
| Fase C · generalización | La evaluación está implementada, pero no constan resultados entrenados con FF++. | Completar test in-domain y cross-manipulation con los mismos checkpoints y splits. |
| Entorno y demo | El runtime real de Kaggle y sus versiones no están registrados en la documentación revisada. | Guardar GPU, `pip freeze`, semilla y commit con las corridas; verificar ejecución limpia de notebooks. |
| Evidencias E2 | El documento Word/Google Docs de evidencias aún no existe. | Prepararlo y entregarlo; registrar su enlace en [E2](E2_Tareas_programacion.md). Se puede documentar ahora la evidencia existente y actualizarla tras las corridas FF++. |
| Informe | Plan y estructura de tareas definidos; resultados experimentales principales pendientes. | Redactar resultados/conclusiones solo después de las corridas; si no concluyen, declarar la limitación y evidencia disponible. |

## Secuencia sugerida hasta la entrega

| Fecha objetivo | Hito de salida |
|---|---|
| 27–29 oct | Preparar sesión Kaggle, revisar acceso/dataset, correr preprocesamiento y validar conteos/recortes. |
| 30 oct–3 nov | Ejecutar BCE y Focal γ=2; guardar configuración, versiones, métricas y checkpoints fuera de Git cuando corresponda. |
| 4–6 nov | Completar cross-manipulation; generar tablas y diagnóstico Grad-CAM. Correr γ=1/5 o una semilla adicional solo si el núcleo está estable. |
| 7–9 nov | Cerrar demo reproducible, verificar notebooks y preparar informe con resultados y limitaciones. |
| 10 nov | Revisión cruzada, limpieza de salidas protegidas, revisión final de formato y repositorio. |
| 11 nov | Entrega E3. |

## Extensiones: después del experimento principal

Estas extensiones amplían el alcance; no deben bloquear la comparación mínima BCE vs. Focal γ=2 ni la tabla principal cross-manipulation.

1. **Barrido de γ ∈ {0, 1, 2, 5} y semillas adicionales** — primera extensión recomendada, porque explora directamente la intervención central. Añadir γ=1/5 y más semillas solo si las corridas principales terminan y pueden compararse bajo el mismo protocolo.
2. **Estrés por compresión c40** — reutilizar el modelo y pipeline, separar el resultado del objetivo principal c23.
3. **Evaluación cross-dataset (DFDC Preview; respaldo: Celeb-DF v2)** — requiere preparar datos, homologar etiquetas/protocolo y revisar acceso/licencia; intentar solo con tiempo suficiente.
4. **Rama de alta frecuencia (SRM)** — extensión exploratoria de arquitectura, con hipótesis y comparación propia. Es la de menor prioridad por el costo de implementación y entrenamiento.

Al iniciar una extensión, registrar antes su hipótesis, alcance y criterio de éxito en `DECISIONS.md`; si no se completa, marcarla como pendiente sin presentarla como resultado.

## 1. Qué cuenta como demo reproducible

Una sesión/notebook de Kaggle que pueda ejecutarse con los datos autorizados y las instrucciones del repositorio, y que genere artefactos identificables: configuración, semilla, hash de commit, versiones, métricas y figuras/tablas. Los datos, videos, recortes y checkpoints grandes o protegidos no se suben a GitHub. Se documenta qué partes se ejecutaron y cualquier limitación de acceso o cómputo.

## 2. Prioridades de trabajo

| Prioridad | Trabajo | Criterio mínimo |
|---|---|---|
| **P0** | Reproducir preprocesamiento y entrenamiento comparativo BCE (γ=0) / Focal (γ=2) de las fases B y C | Mismo split, semilla y configuración; salidas y métricas verificables en FF++ c23 autorizado. |
| **P0** | Completar evaluación cross-manipulation | AUC por frame/video en Deepfakes, Face2Face, FaceSwap y NeuralTextures; tabla comparativa y lectura honesta. |
| **P0** | Cerrar demo, documentación y reporte | Instrucciones probadas, artefactos reproducibles y texto de 4–8 páginas alineado con los resultados disponibles. |
| **P1** | Barrido γ ∈ {0, 1, 2, 5} y diagnósticos del mecanismo | Añadir γ=1 y γ=5, comparar semillas/curvas y porcentaje de ejemplos fáciles si el tiempo de GPU alcanza. |
| **P1** | Teardown en Kaggle y Grad-CAM | GPU/latencia/memoria y mapas sobre ejemplos FF++; interpretar sin atribuir causalidad. |
| **P2** | Evaluación c40 y cross-dataset (DFDC Preview; respaldo Celeb-DF v2) | Ejecutar solo si se dispone de acceso y tiempo; reportar como estrés complementario. |

## 3. Lista de tareas

### A · Teardown y demo de la arquitectura

- [ ] Ejecutar `01_teardown_xception.ipynb` de arriba abajo en el entorno Kaggle objetivo; registrar GPU, versiones y commit.
- [x] Versionar tablas/figuras de teardown CPU disponibles en `results/teardown/`.
- [ ] Ejecutar las mediciones que dependen de GPU y datos: latencia/memoria en GPU, activaciones/ERF sobre caras y Grad-CAM; limpiar salidas de caras antes de exportar el notebook.
- [ ] Actualizar las lecturas de tablas/figuras y distinguir resultados de Xception preentrenada de resultados del detector ajustado con FF++.

### B · Intervención BCE vs. Focal Loss

- [ ] Ejecutar `02_preprocesamiento_ffpp.ipynb` usando acceso autorizado al Dataset privado; verificar conteos, particiones por video y recortes de 16 frames.
- [ ] Revisar grillas de recortes y tasa de frames sin cara; registrar cualquier descarte o desviación del protocolo.
- [ ] Ejecutar `03_entrenamiento_bce_vs_focal.ipynb` para BCE/γ=0 y Focal/γ=2 con los mismos datos, split, semilla y demás hiperparámetros.
- [ ] Guardar para cada corrida configuración, semilla, commit, versiones del entorno, historial de pérdida/AUC y checkpoint como artefacto local de Kaggle; conservar en el repositorio solo salidas permitidas y livianas.
- [ ] Ejecutar y registrar las pruebas de pérdidas en un runtime con PyTorch.
- [ ] Comparar AUC in-domain por frame y por video; reportar accuracy/confusión como secundarias y no inferir mejora por fluctuaciones pequeñas.
- [ ] Añadir γ=1 y γ=5 según presupuesto; comparar el porcentaje de ejemplos fáciles por época y, si alcanza, repetir con semillas adicionales.

### C · Robustez bajo cambios de distribución

- [ ] Completar evaluación en Face2Face, FaceSwap y NeuralTextures con el test oficial completo y las caras originales correspondientes.
- [ ] Preparar una tabla única de AUC por frame/video para BCE, γ=2 y los γ opcionales; incluir caída relativa frente al in-domain y cantidad de videos evaluados.
- [ ] Generar/revisar Grad-CAM BCE vs. Focal en selección emparejada de aciertos y errores; registrar IDs sin publicar imágenes de FF++.
- [ ] Evaluar c40 y DFDC Preview (o Celeb-DF v2 como respaldo) solo si el acceso, formato y cómputo lo permiten; separar estos resultados del objetivo principal.
- [ ] Interpretar si los datos apoyan o refutan la hipótesis; indicar límites del tamaño del subconjunto, semillas, compresión y dataset.

### Demo, trazabilidad y limpieza

- [ ] Confirmar que un integrante distinto puede seguir README/notebooks y producir una corrida corta sin rutas locales personales.
- [ ] Verificar que las salidas incluyen semilla, configuración, hash de commit y versiones (`pip freeze` o archivo de entorno equivalente).
- [ ] Ejecutar `Restart & Run All` para cada notebook que se presente como reproducible; documentar celdas que dependan de secretos, descarga autorizada o artefactos de sesión.
- [ ] Ejecutar `python tools/limpiar_caras.py` y `python tools/limpiar_caras.py --check` antes de versionar notebooks ejecutados.
- [ ] Revisar `.gitignore`: no incluir FF++, recortes, credenciales, checkpoints pesados ni salidas de Kaggle temporales.
- [ ] Actualizar `README.md`, `docs/DECISIONS.md` y [E2](E2_Tareas_programacion.md) con estado final, comandos, decisiones y enlaces a artefactos versionados.
- [ ] Elaborar y registrar el enlace al documento de evidencias E2; reutilizar su procedimiento y evidencia como insumo, sin duplicar el reporte dentro del informe final E3.

### Informe WAIMLAp (4–8 páginas)

- [ ] Alinear título, autores/afiliación y resumen con el alcance real del experimento.
- [ ] Describir contexto, problema, pregunta, metodología (datos, splits, protocolo, modelo, pérdidas y métricas) con precisión suficiente para reproducir.
- [ ] Presentar resultados disponibles en tablas/figuras; etiquetar como pendiente cualquier ejecución que no haya concluido.
- [ ] Discutir resultados positivos o nulos, amenazas a validez y limitaciones; cerrar con conclusiones respaldadas por los datos.
- [ ] Verificar referencias, formato de plantilla, legibilidad de figuras y extensión de 4–8 páginas.
- [ ] Hacer una revisión cruzada: cada integrante explica una parte del método y verifica que el texto no exceda la evidencia.

## 4. Decisión de organización: notebook y código Python

Se recomienda mantener módulos Python independientes cuando contengan lógica reutilizable o testeable (lectura/validación de splits, preprocesamiento, dataset, pérdidas, entrenamiento, métricas y Grad-CAM). Los notebooks deben orquestar el flujo, documentar decisiones y mostrar resultados; no conviene duplicar allí toda la implementación. Esto mejora pruebas, reduce divergencia entre notebooks y CLI y permite importar la misma lógica en Kaggle.

No todo script CLI necesita ser parte de la demo E3: si una funcionalidad no se usa, prueba ni documenta en el flujo entregado, puede quedar como pipeline complementario claramente identificado o proponerse para retirar después de revisar dependencias. Mantener dos rutas activas con protocolos distintos requiere documentar cuál produce los resultados oficiales; para E3, la ruta de referencia sigue siendo `notebooks/kaggle/01–03` + `src/kaggle_e2/`.

## 5. PyTorch frente a TensorFlow

El proyecto ya implementa Xception, `Dataset`/`DataLoader`, BCE/Focal Loss, entrenamiento y evaluación con PyTorch; `timm` aporta `legacy_xception` y pesos preentrenados compatibles. Mantener PyTorch evita reescribir y volver a validar el pipeline, los notebooks Kaggle y las pruebas. TensorFlow no ofrece una ventaja necesaria para la pregunta experimental actual; cambiar de framework cerca de E3 introduciría trabajo y riesgo sin mejorar la comparación controlada. Esta decisión responde al stack y objetivo del proyecto, no a que TensorFlow sea incapaz de realizarlo.

## 6. Criterio de cierre E3

- [ ] Demo reproducible documentada, ejecutable con acceso autorizado a datos y sin afirmaciones basadas en resultados sintéticos como si fueran FF++.
- [ ] Comparación principal BCE vs. γ=2 y evaluación cross-manipulation presentadas o, si no fue posible correrlas, limitación y evidencia disponible declaradas explícitamente.
- [ ] Informe de 4–8 páginas revisado, consistente con `docs/DECISIONS.md` y las salidas guardadas.
- [ ] Repositorio limpio de datos protegidos, credenciales y artefactos temporales; cambios revisados y guardados en Git.

## 7. Registro de avance

| Fecha | Cambio/ejecución | Evidencia o limitación |
|---|---|---|
| 2026-10-27 | Creación del plan E3 a partir de las fases A–C y el estado documentado de E2. | La implementación de notebooks está presente; las corridas reales FF++ siguen pendientes según `docs/DECISIONS.md`. |
