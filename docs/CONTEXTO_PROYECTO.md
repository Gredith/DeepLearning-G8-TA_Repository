# Contexto del proyecto · Grupo 8 · Deep Learning PUCP 2026-2

> Documento de contexto para un asistente (Claude) que ayude a redactar o corregir entregables del proyecto: informe, extended abstract, póster o presentaciones.
> Fuente de verdad: el repositorio del grupo (`docs/DECISIONS.md`, `docs/E2_Tareas_programacion.md`, `results/teardown/`). Estado al **21/10/2026**.

---

## 0. Reglas para el asistente

1. **No inventes resultados.** Todavía **no hay** AUC, accuracy ni Grad-CAM de modelos entrenados con FF++. Las únicas cifras reales son las de la Fase A (sección 6). Si un documento pide resultados de entrenamiento, deja el espacio marcado como pendiente.
2. **Usa los valores de este documento, no los de la propuesta E1 original.** La E1 (7/10) decía «20–30 frames», «MTCNN o RetinaFace», «224×224 o 299×299», «T4/P100» y «200–300 videos por clase». **Todo eso quedó superado.** Los valores vigentes están en la sección 4.
3. Los entregables al workshop (abstract y póster) se escriben **en inglés**. Los del curso (E1, E2, cuaderno de decisiones) se escriben **en español**.
4. Escribe γ (gamma), no «y». Escribe «γ ∈ {0, 1, 2, 5}», nunca «γ ≤ {…}».
5. Hipótesis y resultados se cuentan con honestidad: si la intervención no mejora, eso es un resultado válido y se reporta.
6. Licencia: no se redistribuyen videos, frames ni recortes de FF++, y no se enlaza el script oficial de descarga.

---

## 1. Datos generales

| Campo | Valor |
|---|---|
| Título (ES) | Robustez de detectores de deepfakes: ¿Focal Loss mejora la robustez cross-manipulation de Xception? |
| Título (EN) | Does Focal Loss Improve the Cross-Manipulation Robustness of Xception-Based Deepfake Detectors? |
| Curso | 1INF52 Deep Learning, horario H-1081, PUCP, Facultad de Ciencias e Ingeniería |
| Profesor | Dr. César Armando Beltrán Castañón |
| Equipo | Grupo 8 |
| Integrantes | Alvarez Castillo, Diego Stefano (20220461) · Fernandez Ccerhuayo, Yaser Andre (20222959) · Segura Mamani, Ronaldo Tito (20222051) |
| Autor de correspondencia | a20220461@pucp.edu.pe |
| Workshop | WAIMLAp 2026 (Workshop on Artificial Intelligence and Machine Learning Applications), 18–20 de noviembre de 2026, Lima, waimlap.org |

### Cronograma

| Fecha | Hito |
|---|---|
| 07 oct | **E1**: propuesta de tema (entregada) |
| 21 oct | **E2**: pipeline, teardown (Fase A), primera corrida focal y cuaderno de decisiones en GitHub |
| 22 oct – 4 nov | Barrido de γ (Fase B) y evaluación bajo shift (Fase C) |
| 11 nov | **E3**: demo reproducible e informe final (4–8 págs., plantilla WAIMLAp) |
| 18–20 nov | **E4**: póster en WAIMLAp |

---

## 2. Problema, pregunta e hipótesis

**Problema.** Los detectores de deepfakes reportan AUC cercanos a 0.99 sobre la manipulación con la que se entrenaron, pero su rendimiento cae mucho ante generadores que no vieron. Esto sugiere que las CNN memorizan artefactos propios del generador (alta frecuencia, bordes de mezcla, texturas de color) en lugar de aprender una noción general de «cara manipulada» (Li et al., 2020; Luo et al., 2021).

**Enfoque.** No se busca solo documentar esa caída, que ya es conocida, sino **usarla como instrumento de medición** para entender qué aprende Xception y probar si **una sola intervención controlada** sobre la pérdida la reduce.

**Pregunta principal.** Un detector Xception entrenado con una sola manipulación, ¿aprende «caras falsas» en general o solo la huella de ese generador? Y si se cambia la pérdida de BCE a focal loss, ¿se reduce la caída frente a manipulaciones no vistas?

**Subpreguntas:**
1. ¿Cuánto cae el AUC al pasar de Deepfakes (entrenamiento) a Face2Face, FaceSwap, NeuralTextures, c40 y DFDC?
2. ¿En qué regiones de la cara se apoya el modelo (Grad-CAM) cuando acierta y cuando falla? ¿Cambia con focal loss?
3. ¿Cómo varía la robustez según γ?

### Hipótesis H-B1 (escrita antes de correr): BCE → Focal (γ = 2), entrenando solo en Deepfakes c23
- **Mecanismo:** con BCE el modelo satura rápido en los ejemplos fáciles, que son los de artefactos más obvios del generador. Focal loss les quita peso y desplaza el gradiente hacia los difíciles: falsificaciones sutiles, frames comprimidos, caras de perfil.
- **Predicciones:**
  1. AUC in-domain casi igual (|Δ| < 0.01).
  2. AUC cross-manipulation algo mayor, **más en FaceSwap** (comparte artefactos de mezcla con Deepfakes) **que en NeuralTextures**.
  3. El % de ejemplos fáciles (p_t > 0.9) crece más lento con γ = 2.
- **Resultado nulo posible:** si los difíciles comparten la misma huella del generador, focal loss solo afinaría esa huella.
- **Qué la refutaría:** que el % de fáciles crezca igual con γ = 2 o que el AUC cross empeore de forma consistente entre semillas.

### Hipótesis H-B2: barrido γ ∈ {0, 1, 2, 5} (después de E2)
- Se espera una relación no monótona. Con γ = 5 el modelo ignora casi todo el train y el AUC in-domain cae.

---

## 3. Diseño en tres fases

| Fase | Contenido |
|---|---|
| **A · Teardown** | Parámetros y FLOPs por bloque y por flow (entry/middle/exit); separable vs. convolución estándar; campo receptivo teórico y **ERF** (effective receptive field); activaciones; Grad-CAM; latencia y memoria |
| **B · Intervención** | BCE → focal loss. Todo lo demás fijo: arquitectura, datos, splits, semilla, épocas y optimizador. Primero BCE vs. γ = 2; luego γ ∈ {0, 1, 2, 5} |
| **C · Estrés y honestidad** | Entrenar solo en Deepfakes c23; evaluar en Face2Face, FaceSwap y NeuralTextures (cross-manipulation), FF++ c40 (compresión) y DFDC Preview (cross-dataset; respaldo: Celeb-DF v2). Diagnóstico de fallos con Grad-CAM |

**Ablación secundaria opcional:** una rama de alta frecuencia (filtros SRM) en el stem, con su propia hipótesis.

---

## 4. Decisiones técnicas vigentes (valores que deben usarse)

| Tema | Decisión | Por qué |
|---|---|---|
| Modelo | **Xception original** (`legacy_xception` de timm), pesos ImageNet, **1 logit** de salida (real = 0, falso = 1) | Es la línea base de FF++. Las `xception41/65/71` de timm son la variante «aligned» de DeepLab, otra arquitectura |
| Entrada | **299×299**, normalización media/desv. **[0.5, 0.5, 0.5]** (`default_cfg` de los pesos) | Entrada nativa. A 224 el mapa final pasaría de 10×10 a 7×7 y cambiaría el ERF |
| Detector facial | **MTCNN** (facenet-pytorch); cara de mayor área; caja cuadrada ampliada **1.3×** (como FF++) | Corre en GPU en Kaggle. El margen 1.3× incluye el borde de mezcla, donde suelen estar los artefactos |
| Frames | **16 frames equiespaciados por video** (PNG); los frames sin cara se registran y se omiten | Presupuesto de GPU. Los frames consecutivos son casi redundantes |
| Splits | **Oficiales de FF++ por video**: 360/70/70 pares (720/140/140 IDs). Cada par [a, b] lleva a, b, a_b y b_a al mismo split; ninguna identidad cruza particiones (verificado con `assert`) | Partir por frame filtraría información e inflaría el AUC |
| Subconjunto | **100 pares de train, 30 de val y los 70 de test completos** (determinista, semilla 42). Train: 200 videos reales + 200 falsos × 16 frames = 6400 imágenes | Kaggle da ~30 h/semana de GPU por cuenta. El test completo hace comparables las evaluaciones cross |
| Compresión | c23 (c40 solo para estrés) | — |
| Entrenamiento | Adam, **lr 2×10⁻⁴**, batch 32, **1 época solo cabeza (warm-up) + 5 épocas red completa**, AMP fp16, flip horizontal como única aumentación, mejor checkpoint por AUC de validación, semilla 42, cuDNN determinista | Protocolo de FF++. «Una hipótesis, un cambio, una medición» |
| Focal loss | FL(p_t) = −(1 − p_t)^γ · log(p_t), con **α = None** (sin ponderación por clase) | Train está balanceado 1:1. Con α = None y γ = 0 la pérdida es **exactamente BCE**, lo comprueban tests unitarios. Así se aísla el efecto de γ |
| Métricas | Principal: **AUC-ROC por frame y por video** (promedio de probabilidades por video). Secundarias: accuracy (umbral 0.5) y matriz de confusión, caída relativa de AUC = (AUC_in − AUC_cross)/AUC_in, latencia, FLOPs | El AUC no depende del umbral. Focal tiende a dar probabilidades menos extremas |
| Diagnóstico del mecanismo | **% de ejemplos «fáciles» (p_t > 0.9) por época** | Distingue «funciona por el mecanismo» de «funciona por azar» |
| Grad-CAM | Capa **`block12`** (último bloque residual con mapa espacial, 1024×10×10). Grilla fija: 8 reales y 8 falsas bien clasificadas, 8 falsos negativos y 8 falsos positivos; imagen · BCE · focal | Su ERF al 50 % (~140 px) todavía localiza regiones de la cara |
| Cómputo | Kaggle Notebooks, **GPU T4**, 3 cuentas (~30 h/semana c/u), dataset privado compartido | — |

> **Nota sobre las dos rutas de código.** Hay notebooks E2 (`notebooks/kaggle/01–03` + `src/kaggle_e2/`), que son los que producen los resultados, y un pipeline CLI (`src/train.py`, `src/data_pipeline/`, etc.) con 25 frames JPG, lr 1×10⁻⁴, 10 épocas y Grad-CAM en `conv4`. **En documentos, usa los valores de la tabla de arriba, que son los de los notebooks.**

---

## 5. Datos

| Dataset | Rol |
|---|---|
| FaceForensics++ c23: original + Deepfakes | Entrenamiento y validación in-domain |
| FF++ c23: Face2Face, FaceSwap, NeuralTextures | Evaluación cross-manipulation |
| FF++ c40 | Estrés por compresión (opcional, prioridad baja) |
| DFDC Preview (Meta) | Cross-dataset (opcional, prioridad baja) |
| Celeb-DF v2 | Respaldo cross-dataset |

El acceso académico a FF++ (TU München) está aprobado y se verificó la descarga de una muestra en Kaggle.

---

## 6. Resultados disponibles: Fase A (teardown, pesos ImageNet, sin datos)

Archivos: `results/teardown/*.csv`.

| Medición | Resultado |
|---|---|
| Tamaño total | **20.8 M parámetros**; **16.7 GFLOPs** (8.4 GMACs) a 299×299 |
| Parámetros por flow | stem 0.1 % · entry 5.2 % · **middle 62.1 %** · exit 32.6 % |
| FLOPs por flow | stem 5.0 % · entry 27.2 % · **middle 55.6 %** · exit 12.1 % |
| Separable → estándar 3×3 | 177.0 M parámetros (**×8.5**) y 137.6 GFLOPs (**×8.2**) |
| RF teórico | 19 px (`block1`), 91 px (fin del entry, `block3`), **283 px en `block5`** (ya supera el recorte de 299), 1083 px a la salida (`conv4`) |
| ERF, lado al 50 % de la masa | 25 px (`block3`), **43–91 px en el middle flow**, 137 px (`block12`), ~157 px (salida) |
| Memoria | pesos 79.4 MB (fp32); activaciones ~282 MB por imagen (cota, fp32); ~9 GB con batch 32 → justifica AMP |
| Latencia CPU (2 hilos) | **131 ms/imagen** (batch 1, mediana); 169 ms/imagen con batch 8 |

**Lectura.** Dos rasgos estructurales apoyan la idea de que Xception aprende huellas locales del generador:
- La convolución depthwise filtra cada canal por separado.
- El ERF del middle flow cubre solo ~15–30 % del lado del recorte, aunque el RF teórico ya lo supera. La red usa información mucho más local de lo que su RF permitiría.

Esto **no prueba** nada sobre el modelo entrenado. Es una predicción que se pondrá a prueba con Grad-CAM.

---

## 7. Estado y pendientes (21/10/2026)

**Hecho:**
- Acceso a FF++ y muestra verificada en Kaggle.
- Splits oficiales descargados y versionados, con test de fuga aprobado.
- Fase A completa en todo lo que no necesita datos (sección 6).
- Pipeline de datos, entrenamiento y evaluación (BCE y focal; AUC por frame y por video) implementado y **probado de punta a punta con datos sintéticos**.
- Tests de focal loss: γ = 0 ≡ BCE, fácil vs. difícil, sin NaN con logits ±50, gradientes finitos.

**Pendiente:**
- Corridas reales en Kaggle: **B-01 `bce_s42`** (γ = 0) y **B-02 `focal_g2_s42`** (γ = 2).
- Tabla cross-manipulation; Grad-CAM BCE vs. focal; latencia y memoria en GPU T4; activaciones y ERF sobre caras.
- Resto del barrido (γ = 1 y γ = 5), 2–3 semillas, c40 y DFDC, según el tiempo disponible.
- Fijar versiones del entorno (`pip freeze` de Kaggle).

**Tabla de resultados que hay que llenar** (entrenando en Deepfakes c23; AUC):

| Pérdida | DF (in-domain) | F2F | FS | NT | c40 | DFDC |
|---|---|---|---|---|---|---|
| BCE (γ = 0) | – | – | – | – | – | – |
| FL γ = 1 | – | – | – | – | – | – |
| FL γ = 2 | – | – | – | – | – | – |
| FL γ = 5 | – | – | – | – | – | – |

Si hay una sola semilla, se indica. Con diferencias pequeñas no se sacan conclusiones.

---

## 8. Entregables y su estado

| Archivo (`docs/`) | Qué es | Estado |
|---|---|---|
| `E1 _Propuesta de tema_Grupo_8.md` y `E1_DeepLearning_Grupo8.pdf` | Propuesta y presentación E1 (7/10) | Histórico; contiene valores ya superados (ver regla 2) |
| `E2_Tareas_programacion.md` | Plan de tareas E2 con casillas | Vivo |
| `DECISIONS.md` | Cuaderno de decisiones (D-01…D-11), hipótesis, registro de experimentos y uso de IA | Vivo, fuente de verdad |
| `WAIMLAp2026_Extended_Abstract_Grupo8_v2.docx` | Extended abstract (EN, 2 págs.): Introduction, Methodology, Results and Discussion, Conclusions, References [1]–[8] | Alineado al 21/10; resultados de la Fase B pendientes |
| `WAIMLAp2026_Poster_Grupo8_v2.pptx` | Póster A0 vertical (EN): Motivation, Objectives, Methodology (+ Fig. 1 pipeline), Results & Discussion (+ Table 1 y recuadro Grad-CAM), Conclusions, References, Acknowledgments, QR | Alineado al 21/10; faltan la Tabla 1, la figura Grad-CAM y el QR |

**Take-home del póster:** «In-domain AUC says little about the real world — measure the drop, then explain it.»

**Pregunta para discutir en el póster:** ¿basta con re-ponderar los ejemplos difíciles, o hay que forzar explícitamente al detector a mirar más allá de la huella del generador?

**Limitaciones que hay que mencionar:**
- Los generadores de FF++ son anteriores a los de difusión.
- Posible sesgo demográfico (edad, género, tono de piel).
- Cómputo limitado a GPUs gratuitas de Kaggle, de ahí el subconjunto de entrenamiento.
- Una sola semilla al inicio.

---

## 9. Uso responsable

- **Datos:** rostros de personas reales, solo para investigación académica. No se redistribuyen, no se intenta identificar a nadie y no se generan deepfakes nuevos. FF++ va en un Kaggle Dataset privado. En el repositorio público solo hay código, splits (metadatos públicos) y tablas numéricas, sin figuras con caras.
- **IA:** se usó Claude para explorar temas, redactar, generar código (teardown, pipeline, entrenamiento, focal loss, Grad-CAM) y revisar el avance. El grupo revisó y decidió cada punto. El detalle está en `DECISIONS.md`.

---

## 10. Agradecimientos

- Curso 1INF52 Deep Learning, PUCP; Prof. César Armando Beltrán Castañón.
- Autores de FaceForensics++ (TU München) por el acceso académico; Meta por DFDC Preview.
- Cómputo: Kaggle Notebooks (GPU T4).

---

## 11. Referencias

**Núcleo (las que usan el abstract y el póster):**
1. A. Rössler, D. Cozzolino, L. Verdoliva, C. Riess, J. Thies, M. Nießner, «FaceForensics++: Learning to detect manipulated facial images», ICCV 2019, pp. 1–11.
2. F. Chollet, «Xception: Deep learning with depthwise separable convolutions», CVPR 2017, pp. 1251–1258.
3. T.-Y. Lin, P. Goyal, R. Girshick, K. He, P. Dollár, «Focal loss for dense object detection», ICCV 2017, pp. 2980–2988.
4. R. R. Selvaraju et al., «Grad-CAM: Visual explanations from deep networks via gradient-based localization», ICCV 2017, pp. 618–626.
5. L. Li et al., «Face X-ray for more general face forgery detection», CVPR 2020, pp. 5001–5010.
6. Y. Luo et al., «Generalizing face forgery detection with high-frequency features», CVPR 2021, pp. 16317–16326.
7. A. Haliassos et al., «Lips don't lie: A generalisable and robust approach to face forgery detection», CVPR 2021, pp. 5039–5049.
8. B. Dolhansky et al., «The Deepfake Detection Challenge (DFDC) preview dataset», arXiv:1910.08854, 2019.
9. W. Luo et al., «Understanding the effective receptive field in deep convolutional neural networks», NeurIPS 2016 (base del ERF).

**Complementarias (propuesta E1):**
- Verdoliva (2020), «Media forensics and DeepFakes: An overview», IEEE JSTSP.
- Y. Li et al. (2020), «Celeb-DF», CVPR.
- Zhang et al. (2016), «MTCNN», IEEE SPL.
- Du et al. (2020), «Locality-aware autoencoder», CIKM.
- Xu et al. (2022), «Supervised contrastive learning for generalizable deepfake detection», WACVW.
- Fridrich y Kodovsky (2012), «SRM», IEEE TIFS.
- Zhou et al. (2018), «Rich features for manipulation detection», CVPR.
- Brodarič et al. (2024), «Cross-dataset deepfake detection», CVWW.
- Amada et al. (2025), arXiv:2507.22601.
- Tan y Le (2019), «EfficientNet», ICML.
- Rana et al. (2026), ICBDML.
- Jeswani y Priyadharshini (2026), ICITIIT.
