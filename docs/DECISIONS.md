# Cuaderno de decisiones — Grupo 8

**Proyecto:** Robustez de detectores de deepfakes: ¿Focal Loss mejora la robustez cross-manipulation de Xception?
**Curso:** 1INF52 Deep Learning, PUCP 2026-2 · Dr. César A. Beltrán Castañón
**Integrantes:** Alvarez Castillo, Diego Stefano · Fernandez Ccerhuayo, Yaser Andre · Segura Mamani, Ronaldo Tito

Cada entrada registra **qué** se decidió, **qué alternativas** había, **por qué** y **dónde hubo asistencia de IA**. Antes de cada experimento se escribe la hipótesis previa (sección "Hipótesis") y después se contrasta con el resultado (sección "Registro de experimentos"). Todos los integrantes deben poder defender cualquier entrada.

> **Estado al 21/10/2026 (E2).**
> - El repositorio tiene dos rutas de ejecución que comparten splits y modelo:
>   - **Notebooks E2** (`notebooks/kaggle/01–03` + `src/kaggle_e2/`): son los que producen los resultados de este avance (subconjunto por presupuesto, D-05).
>   - **Pipeline CLI** (`src/data_pipeline/`, `src/train.py`, `src/evaluate*.py`, `src/teardown/`): pipeline completo con manifiestos y procedencia, para las corridas de la Fase B completa.
> - Fase A: completa para todo lo que no depende de datos (parámetros, FLOPs, RF, separable vs. estándar, latencia y memoria en CPU). Pendiente en Kaggle: latencia y memoria en GPU, activaciones sobre caras, ERF sobre caras y Grad-CAM.
> - Fase B: pipeline de los notebooks probado de punta a punta con datos sintéticos. La primera corrida real BCE vs. Focal (γ=2) se ejecuta en Kaggle (notebook 03).

## Índice de decisiones

Las decisiones vigentes y su justificación se detallan en D-01–D-11. Esta sección se omite para evitar duplicar el resumen de cada entrada; para el estado de tareas y evidencia consulta [E2](E2_Tareas_programacion.md) y [E3](E3_Tareas_programacion.md).

---

## Decisiones

### D-01 · Modelo base: Xception (`legacy_xception`, timm) preentrenada en ImageNet
- **Alternativas:** EfficientNet-B4 (soluciones ganadoras del DFDC), ViT/DINOv2.
- **Por qué:** es la línea base del propio benchmark FaceForensics++ (Rössler et al., 2019), así que hay números publicados para validar la reproducción. Además, su núcleo (separables + residuales) se puede diseccionar con lo visto en el curso. `legacy_xception` es la Xception original de Chollet (2017) con los pesos que usa FF++; las `xception41/65/71` de timm son la variante "aligned" de DeepLab, que es otra arquitectura.
- **Respaldo:** EfficientNet-B4 si otro grupo elige el mismo artículo.
- **IA:** el asistente sugirió distinguir `legacy_xception` de las variantes aligned; el grupo lo verificó revisando la arquitectura impresa.

### D-02 · Entrada 299×299, normalización media 0.5 / desviación 0.5
- **Alternativa:** 224×224, más barata.
- **Por qué:** es la entrada nativa de Xception y la configuración de los pesos (`default_cfg`). A 224 el mapa final pasaría de 10×10 a 7×7 y cambiaría el ERF medido en la Fase A. Preferimos no introducir un cambio extra respecto a la referencia.

### D-03 · Recorte de cara: MTCNN, caja ampliada ×1.3, cuadrada, 16 frames equiespaciados por video
- **Alternativas:** dlib (lo que usó FF++), RetinaFace, Haar cascade; 270 frames por video como en FF++.
- **Por qué:** MTCNN (facenet-pytorch) corre en GPU en Kaggle y se instala con `--no-deps` sin tocar la versión de torch. El factor ×1.3 es el de FF++: incluye el borde de mezcla, que es justamente donde suelen estar los artefactos. Los 16 frames equiespaciados se eligen por presupuesto de GPU (D-05). Los frames consecutivos son casi redundantes, así que muestrear espaciado da más diversidad por frame.
- **Riesgo registrado:** si no se detecta cara, el frame se descarta. Se cuenta por video en `crops/manifest.csv`.

### D-04 · Partición POR VIDEO con los splits oficiales de FF++ (360/70/70 pares)
- **Alternativa descartada:** partición aleatoria por frame.
- **Por qué:** los frames de un mismo video son casi idénticos. Partir por frame filtraría información y daría un AUC inflado. Cada par oficial [a, b] lleva al mismo split los reales `a`, `b` y los falsos `a_b`, `b_a`, así que ninguna identidad cruza splits. El notebook 02 lo verifica con un `assert` y el pipeline CLI con `assert_disjoint_splits`.

### D-05 · Subconjunto por presupuesto de GPU: 100 pares de train, 30 de val y los 70 de test completos
- **Alternativa:** los 360 pares de train.
- **Por qué:** Kaggle da ~30 h/semana de GPU por cuenta, y el barrido de γ necesita al menos 4 corridas × varias semillas. Con 100 pares (200 videos reales + 200 falsos × 16 frames = 6400 imágenes) una corrida dura minutos. **El test se mantiene completo** para que las comparaciones cross-manipulation sean sobre los mismos 70 pares oficiales. El subconjunto es determinista (semilla 42).
- **Consecuencia:** el AUC in-domain puede quedar por debajo del reportado en FF++ (que usa 720 videos de train y 270 frames por video). Se reportará esa brecha (Fase C, honestidad).

### D-06 · Protocolo de entrenamiento idéntico entre BCE y Focal
- Adam, lr 2e-4 (FF++); batch 32; 1 época solo cabeza (warm-up, como FF++) + 5 épocas red completa; AMP fp16; flip horizontal como única augmentación; mejor checkpoint por AUC de validación; semilla 42 con cuDNN determinista.
- **Por qué:** "una hipótesis, un cambio, una medición". Lo único que varía entre corridas es γ.
- **IA:** el código de `src/kaggle_e2/train.py` lo generó el asistente; el grupo revisó el bucle (`zero_grad → forward → loss → backward → step`), el cálculo de AUC por video y el congelamiento de la cabeza en el warm-up.

### D-07 · Focal Loss sin ponderación por clase (α = None)
- **Alternativa:** α = 0.25, el valor de Lin et al. (2017).
- **Por qué:** α corrige el desbalance de clases. En detección densa el fondo domina 1000:1, pero aquí train es 1:1 (cada par aporta 2 reales y 2 falsos). Si cambiáramos α y γ a la vez, no sabríamos cuál de los dos causó el efecto. Con α = None y γ = 0 la pérdida es **exactamente BCE**: lo comprueban `tests/test_losses.py` (implementación de `src/models/`) y `tests/test_kaggle_e2_losses.py` (implementación de los notebooks). Así la comparación aísla el efecto de γ.

### D-08 · Diagnóstico del mecanismo: % de ejemplos "fáciles" (p_t > 0.9) por época
- **Por qué:** la hipótesis dice que Focal Loss actúa reduciendo el peso de los ejemplos fáciles. Medir solo el AUC final no distingue "funciona por ese mecanismo" de "funciona por azar". Registrar la fracción de fáciles por época permite verificar si el mecanismo ocurre, aunque el AUC no cambie.

### D-09 · Métricas: AUC a nivel de frame y de video (promedio de probabilidades por video)
- **Por qué:** el AUC no depende del umbral, lo que importa cuando cambia la distribución (cross-manipulation). El nivel de video es el que usa FF++/DFDC. La accuracy a 0.5 se reporta solo como secundaria, porque Focal Loss tiende a producir probabilidades menos extremas y eso mueve el umbral óptimo.

### D-10 · Grad-CAM sobre `block12`
- **Alternativa:** `conv4`, la última capa (la que usa `src/teardown/gradcam.py` del pipeline CLI).
- **Por qué:** `block12` es el último bloque residual con mapa espacial (1024×10×10) antes de las separables finales. En la Fase A su ERF al 50 % es de ~140 px, frente a un recorte de 299 px, así que todavía localiza regiones de la cara.

### D-11 · Licencia de FaceForensics++ en el repositorio
- **Qué:** el repositorio es público, así que no se versionan videos, frames, recortes de caras ni el script oficial de descarga (tampoco su URL).
  - El script se sube a Kaggle como **Dataset privado**; los notebooks lo encuentran en `/kaggle/input/`.
  - Las celdas que muestran caras llevan el tag `caras`. `python tools/limpiar_caras.py` borra sus salidas antes de versionar un notebook ejecutado, y `--check` falla si queda alguna.
  - Las figuras con caras (`ejemplo_recortes.png`, `fig_gradcam_*`, `fig_activaciones_*`) están en `.gitignore`. Sí se versionan tablas y métricas numéricas.
  - Los splits de `results/splits/` son metadatos públicos del repositorio `ondyari/FaceForensics`, no datos del dataset.
- **Por qué:** los términos de uso de FF++ no permiten redistribuir el dataset ni el script de descarga.
- **Incidente:** el 21/10 se detectó que el notebook explorador tenía dos figuras con frames de FF++ y la URL del script. Se eliminaron y se reescribió el historial de Git para quitarlas también de los commits anteriores.
- **IA:** el asistente detectó el problema durante una revisión del avance y propuso el mecanismo de tags; el grupo decidió reescribir el historial.

---

## Hallazgos de la Fase A (teardown) — `notebooks/kaggle/01_teardown_xception.ipynb`

| Medición | Resultado | Implicación para el proyecto |
|---|---|---|
| Tamaño | 20.8 M parámetros; 16.7 GFLOPs (8.4 GMACs) a 299×299 | Coincide con lo publicado para Xception; viable en T4 con AMP |
| Parámetros por flow | stem 0.1 %, entry 5 %, middle 62 %, exit 33 % | El núcleo repetido (8 bloques de 728 canales) concentra la capacidad |
| FLOPs por flow | stem 5 %, entry 27 %, middle 56 %, exit 12 % | La alta resolución cuesta mucho cómputo por parámetro |
| Separable → estándar | 177 M parámetros (×8.5), 138 GFLOPs (×8.2) | El ahorro ~9× por capa permite tener 36 capas; el sesgo es hacia textura local por canal |
| RF teórico | 91 px al final del entry; 283 px en `block5`; 1083 px a la salida | Desde el middle flow supera el recorte |
| ERF (50 % de la masa) | 25 px (`block3`), 43–91 px (middle), ~157 px (salida) | La red usa información mucho más local que su RF teórico |
| Memoria | pesos 79 MB; activaciones ~282 MB por imagen (fp32, cota) | El batch lo limitan las activaciones, no los pesos → AMP |
| Latencia CPU (2 hilos) | 131 ms/imagen (batch 1, mediana; `results/teardown/latencia_cpu.csv`) | GPU: pendiente en Kaggle |

**Conexión con la pregunta de investigación.** Hay dos rasgos estructurales que apoyan la idea de que Xception aprende huellas locales del generador:
- La convolución depthwise filtra cada canal por separado.
- El ERF del middle flow cubre solo ~15–30 % del lado del recorte.

Esto no prueba nada sobre el modelo entrenado. Es una predicción que la Fase C pone a prueba con Grad-CAM: si la atención se concentra en bordes de mezcla o en texturas, y si cambia con Focal Loss.

---

## Hipótesis (escritas antes de correr)

### H-B1 · BCE → Focal Loss (γ = 2), entrenando solo en Deepfakes c23
- **Mecanismo esperado:** BCE satura rápido en los ejemplos fáciles, que son los de artefactos más obvios del generador. Focal Loss desplaza el gradiente hacia los difíciles.
- **Predicciones:**
  1. AUC in-domain casi igual (|Δ| < 0.01).
  2. AUC cross-manipulation algo mayor, más en FaceSwap que en NeuralTextures.
  3. El % de fáciles crece más lento con γ = 2.
- **Alternativa:** sin mejora, si los difíciles comparten la misma huella del generador.
- **Qué la refutaría:** que el % de fáciles crezca igual con γ = 2 (el mecanismo no ocurre) o que el AUC cross-manipulation empeore de forma consistente entre semillas.

### H-B2 · Barrido γ ∈ {0, 1, 2, 5} *(siguiente paso, después del E2)*
- **Predicción:** relación no monótona. Valores de γ muy altos (5) ignoran casi todo el train, y el AUC in-domain caerá.

---

## Registro de experimentos

| ID | Fecha | Config | Resultado | ¿Coincide con la hipótesis? |
|---|---|---|---|---|
| A-01 | 20/10 | Teardown, pesos ImageNet, CPU | Ver tabla de la Fase A | — (descriptivo) |
| S-01 | 21/10 | Prueba de humo de `src/kaggle_e2` con recortes sintéticos (CPU, 96 px, 2 épocas) | Entrena, guarda checkpoint y evalúa AUC por frame y video en las 4 manipulaciones | — (validación de código, no es resultado) |
| B-01 | *pendiente* | `bce_s42`: γ=0, 100/30/70 pares, 16 frames, 6 épocas | | |
| B-02 | *pendiente* | `focal_g2_s42`: γ=2, igual a B-01 | | |

---

## Uso de asistentes de IA

| Parte | Qué hizo la IA (Claude) | Qué revisó o decidió el grupo |
|---|---|---|
| Propuesta E1 | Ayudó a explorar temas y a redactar | Elección del tema, pregunta e hipótesis |
| `src/kaggle_e2/teardown.py`, notebook 01 | Generó el código de conteo de FLOPs, RF/ERF y latencia, y ejecutó el notebook | Verificar el RF a mano para `block1` (19 px) y contrastar los parámetros con el paper |
| `src/kaggle_e2/` (`data`, `preprocess`, `train`, `losses`, `gradcam`) | Generó el código y una prueba con datos sintéticos | Revisar la partición por video, la equivalencia γ=0 ↔ BCE y el bucle de entrenamiento |
| Revisión del avance E2 (21/10) | Revisó ambos pipelines, verificó cifras y pruebas, detectó el problema de licencia (D-11) y preparó la integración de los notebooks | Elegir la ruta de ejecución, aprobar la reescritura del historial y ejecutar los commits |
| Este cuaderno | Redactó el borrador de las decisiones | Cada integrante valida las entradas que defenderá |

## Registro de cambios

- 2026-10-21: se creó la estructura inicial del proyecto basada en E2.
- 2026-10-21: se organizaron documentos en `docs/`, el código en `src/` y dependencias en `environment/`; notebooks destinados a Kaggle en `notebooks/kaggle/`.
- 2026-10-21: se eligió MTCNN como detector inicial y se definió el formato CSV de manifiestos para el pipeline T1.
- 2026-10-21: se quitaron del notebook explorador las figuras con frames de FF++ y la URL del script de descarga, también del historial (D-11); se añadió `tools/limpiar_caras.py`.
- 2026-10-21: se integraron los notebooks E2 (`notebooks/kaggle/01–03`, `src/kaggle_e2/`, `results/teardown/`) y este cuaderno; la normalización pasó a 0.5 (D-02).
