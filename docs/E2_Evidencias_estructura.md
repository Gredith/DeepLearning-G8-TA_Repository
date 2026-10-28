# E2 · Estructura del documento de evidencias

> **Plantilla de preparación:** usar esta estructura para completar el documento solicitado en Word o Google Docs después de ejecutar los notebooks. Reemplazar cada campo entre corchetes con datos verificados. No presentar como resultado una salida que no se haya generado.

## Portada

- **Proyecto:** Robustez cross-manipulation de Xception con Focal Loss
- **Curso / entrega:** Deep Learning, Equipo 8 · E2
- **Integrantes:** [nombres]
- **Fecha de entrega:** [fecha]
- **Repositorio GitHub:** [URL]
- **Versión evaluada:** rama/commit [rama y hash]

## 1. Propósito y alcance del avance

Describir en un párrafo qué se propuso completar en E2 y qué partes se ejecutaron realmente. Distinguir implementación, ejecución sintética y experimentos con FF++ autorizado.

**Estado de ejecución al entregar:** [completado/parcial, con breve explicación]

## 2. Entorno y reproducibilidad

| Elemento | Valor registrado |
|---|---|
| Plataforma | Kaggle Notebooks |
| GPU / acelerador | [modelo o CPU] |
| Python | [versión] |
| PyTorch / CUDA | [versiones] |
| Dependencias relevantes | [resumen; adjuntar `pip freeze` si aplica] |
| Rama y commit | [rama / hash] |
| Semillas y configuración | [valores o ruta al archivo] |
| Dataset | FaceForensics++ [versión/compresión/splits, acceso autorizado] |

Indicar qué artefactos permanecen en Kaggle por privacidad/licencia y cuáles se exportaron al repositorio. No incluir credenciales, videos ni recortes de rostros.

## 3. Procedimiento y evidencia por notebook

Para cada notebook, indicar: propósito, pasos que se ejecutaron, parámetros/entradas, salidas generadas, evidencia que se presenta y estado/limitaciones. Añadir capturas o tablas legibles en el documento final; conservar los `.ipynb` enlazados en GitHub.

### 3.1 Notebook 01 · `01_teardown_xception.ipynb`

- **Propósito:** [descripción]
- **Secciones ejecutadas:** [números/títulos]
- **Procedimiento y configuración:** [resumen]
- **Resultados:** [tabla/figura y una breve lectura]
- **Ubicación de artefactos versionados:** `results/teardown/` [ajustar si cambió]
- **Secciones no ejecutadas / limitaciones:** [p. ej., GPU o análisis que requiere caras]

Este notebook analiza principalmente Xception preentrenada; separar esos resultados de cualquier resultado del detector ajustado con FF++.

### 3.2 Notebook 02 · `02_preprocesamiento_ffpp.ipynb`

- **Propósito:** preparar FF++ autorizado para el experimento.
- **Datos/splits y parámetros usados:** [compresión, pares, frames, detector, resolución]
- **Procedimiento ejecutado:** [descarga autorizada, validaciones, extracción y recorte]
- **Conteos:** [videos, frames procesados, caras detectadas/no detectadas por partición y clase]
- **Revisión de calidad:** [grilla inspeccionada, incidencias y decisión]
- **Salidas conservadas en Kaggle:** [rutas de recortes/manifiestos; no subir medios protegidos]
- **Evidencia ligera exportada/versionada:** [tablas, logs resumidos u otros permitidos]
- **Estado/limitaciones:** [detalle]

### 3.3 Notebook 03 · `03_entrenamiento_bce_vs_focal.ipynb`

- **Prerequisito:** indicar la ejecución de Notebook 02 y la ubicación/versión de los recortes usados.
- **Modelos/configuración:** BCE (γ=0) y Focal (γ=2), [semilla, épocas, batch, optimizador, checkpoint elegido].
- **Procedimiento ejecutado:** [resumen del entrenamiento, validación y evaluación].
- **Resultados in-domain:** [AUC por frame y por video; métricas secundarias si se reportan].
- **Resultados cross-manipulation:** [tabla por manipulación, AUC y caída relativa].
- **Diagnóstico:** [Grad-CAM/errores, si se ejecutaron].
- **Artefactos privados de Kaggle:** [rutas o identificadores de corridas; no incluir checkpoints pesados en Git].
- **Resultados permitidos versionados:** [rutas a CSV/JSON/figuras sin rostros].
- **Estado/limitaciones:** [corridas ausentes, semillas, fallos o restricciones].

## 4. Resumen consolidado de resultados

Incluir solo resultados realmente ejecutados. Si una corrida no se completó, dejar «pendiente» y explicar por qué; no inferir valores.

| Experimento | Datos/protocolo | Métrica | Resultado | Evidencia/ruta |
|---|---|---|---|---|
| Teardown CPU | Xception preentrenada | [métrica] | [valor] | [archivo/figura] |
| BCE γ=0 | [completar tras ejecución real] | AUC frame/video | [pendiente] | [ruta] |
| Focal γ=2 | [completar tras ejecución real] | AUC frame/video | [pendiente] | [ruta] |
| Cross-manipulation | [completar tras ejecución real] | AUC por manipulación | [pendiente] | [ruta] |

**Lectura breve:** [comparación BCE/Focal, si los datos la permiten; incluir resultados nulos y cautelas].

## 5. Enlaces y evidencia verificable

- Repositorio: [URL GitHub]
- Rama/commit revisado: [URL del commit]
- Notebooks: [enlaces a los notebooks 01, 02 y 03]
- Resultados ligeros en el repositorio: [rutas/enlaces]
- Ejecuciones privadas en Kaggle: [enlaces/identificadores compartibles, si corresponde]

## 6. Pendientes y limitaciones

Enumerar qué no se ejecutó, restricciones de datos o cómputo, diferencias respecto al protocolo y qué se propone para la siguiente fase. Diferenciar claramente los resultados existentes de los pendientes.

## 7. Declaración sobre datos y materiales

Confirmar que no se redistribuyeron videos, frames, recortes, credenciales ni el script oficial de descarga de FF++. Las figuras incluidas deben cumplir la licencia y la política del repositorio.
