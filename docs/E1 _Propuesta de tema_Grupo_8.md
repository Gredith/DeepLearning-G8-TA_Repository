**PONTIFICIA UNIVERSIDAD CATÓLICA DEL PERÚ**

**FACULTAD DE CIENCIAS E INGENIERÍA**

 

**E1 : Propuesta de tema**

 

**Curso:** Deep Learning

**Horario**: H-1081

**Profesor:** Beltrán Castañón, Cesar Armando

**Equipo:** 8

**Integrantes:**

Alvarez Castillo, Diego Stefano (20220461)

Fernandez Ccerhuayo, Yaser Andre (20222959)

Segura Mamani, Ronaldo Tito (20222051)

**Lima, 7 de Octubre 2026**

**Índice**

[**1\. Tema y pregunta de investigación	3**](#1.-tema-y-pregunta-de-investigación)

[**2\. Motivación y relevancia	3**](#2.-motivación-y-relevancia)

[**3\. Artículo y modelo de referencia	3**](#3.-artículo-y-modelo-de-referencia)

[**4\. Datos	4**](#4.-datos)

[**5\. Plan por fases	4**](#5.-plan-por-fases)

[Fase A — Disección instrumentada de Xception	4](#fase-a:-disección-instrumentada-de-xception)

[Fase B — Intervención en el núcleo	4](#fase-b:-intervención-en-el-núcleo)

[Fase C — Estrés y honestidad	5](#fase-c:-estrés-y-honestidad)

[**6\. Métricas, cómputo y riesgos	5**](#6.-métricas,-cómputo-y-riesgos)

[**7\. Cronograma y reparto tentativo	5**](#7.-cronograma-y-reparto-tentativo)

[**8\. Uso responsable de datos y de IA	6**](#8.-uso-responsable-de-datos-y-de-ia)

[**9\. Referencias	6**](#9.-referencias)

## 

# **1\. Tema y pregunta de investigación** {#1.-tema-y-pregunta-de-investigación}

**Título:** Robustez de detectores de deepfakes: ¿Focal Loss mejora la robustez cross-manipulation de Xception?

**Pregunta principal:** Un detector XceptionNet entrenado sobre un solo método de manipulación, ¿aprende a reconocer "caras falsas" en general o solo la huella específica de ese generador? Y si cambiamos la función de pérdida de BCE a focal loss, ¿se reduce la caída de rendimiento frente a manipulaciones no vistas?

**Sub-preguntas:**

* ¿Cuánto cae el AUC al pasar de la manipulación de entrenamiento (Deepfakes) a otras (Face2Face, FaceSwap, NeuralTextures) y a otro dataset (DFDC)?  
* ¿En qué regiones de la cara se apoya el modelo (Grad-CAM) cuando acierta y cuando falla, y cambia eso con focal loss?  
* ¿Cómo varía la robustez cross-manipulation según el parámetro de enfoque γ de focal loss?

La idea no es solo documentar que el detector falla fuera de su dominio, porque eso ya se sabe (L. Li et al., 2020; Luo et al., 2021; Verdoliva, 2020). Usamos esa caída como instrumento de medición para entender qué aprendió el modelo y probar si un cambio controlado en la pérdida la reduce.

# **2\. Motivación y relevancia** {#2.-motivación-y-relevancia}

Los detectores de deepfakes suelen reportar AUC cercanos a 0.99 cuando se evalúan sobre el mismo tipo de manipulación con el que se entrenaron, pero en la práctica un detector desplegado enfrentará generadores que nunca vio. La literatura (Rössler et al., 2019; Dolhansky et al., 2019; L. Li et al., 2020; Haliassos et al., 2021\) muestra que el rendimiento cae fuertemente en ese escenario, lo que sugiere que las CNN se apoyan en artefactos propios de cada generador (patrones de alta frecuencia, bordes de mezcla, texturas) en lugar de en una noción general de “cara manipulada”. Luo et al. (2021) reportan que los detectores CNN tienden a sobreajustarse a texturas de color específicas de cada método, L. Li et al. (2020) se apoyan en los bordes de mezcla, que son comunes a muchas manipulaciones, y Du et al. (2020) muestran que atender a las regiones locales de la falsificación mejora la generalización.

Otras líneas de trabajo abordan la generalización con enfoques distintos al nuestro: usar una imagen registrada de la persona (Amada et al., 2025), evaluar otras familias de modelos sobre FaceForensics++ (Rana et al., 2026; Jeswani y Priyadharshini, 2026\) o comparar detectores modernos entre datasets (Brodarič et al., 2024). Nuestro aporte se limita a una intervención controlada sobre la función de pérdida de un detector CNN clásico.

El tema encaja con el ciclo de tres fases del curso: el modelo base es una CNN (contenido de las unidades ya dictadas: arquitecturas CNN, transfer learning, regularización), la intervención es localizada y medible, y la Fase C de shift de dominio es el centro natural del problema. Además, es computacionalmente viable: clasificación binaria sobre recortes de cara de 224×224.

# **3\. Artículo y modelo de referencia** {#3.-artículo-y-modelo-de-referencia}

**Artículo base:** Rössler, A. et al. (2019). *FaceForensics++: Learning to Detect Manipulated Facial Images.* ICCV 2019\.

**Modelo base:** XceptionNet (Chollet, 2017\) preentrenada en ImageNet y ajustada (fine-tuning) para clasificación binaria real/falso sobre recortes de cara, tal como lo propone FaceForensics++ (Rössler et al., 2019).

**Por qué Xception:**

* Es la línea base de referencia del propio benchmark FF++ (Rössler et al., 2019), por lo que hay números publicados contra los cuales validar nuestra reproducción.  
* Su núcleo (convoluciones separables en profundidad \+ conexiones residuales; Chollet, 2017\) es una CNN clásica que podemos diseccionar con las herramientas vistas en el curso.  
* Hay implementaciones abiertas (timm / PyTorch) con pesos ImageNet.

**Diferenciación con otro equipo:** Nuestro punto de partida es XceptionNet/FF++; como alternativa de respaldo usaríamos EfficientNet-B4 (Tan y Le, 2019), estilo de las soluciones ganadoras del DFDC.

**Objetivo de reproducción:** Obtener in-domain (entrenar y evaluar en Deepfakes c23) un AUC/accuracy cercano al reportado en el artículo (Rössler et al., 2019\) antes de cualquier intervención.

# **4\. Datos** {#4.-datos}

| Dataset | Rol en el proyecto | Acceso |
| :---- | :---- | :---- |
| FaceForensics++ (c23) original \+ Deepfakes | Entrenamiento y validación in-domain | Acceso académico aprobado por los autores (TUM); descarga con el script oficial |
| FF++ Face2Face, FaceSwap, NeuralTextures | Evaluación cross-manipulation (Fase C) | Mismo acceso |
| FF++ c40 | Estrés por compresión (Fase C) | Mismo acceso |
| DFDC Preview (Meta) | Evaluación cross-dataset (Fase C) | Términos de uso de Meta / Kaggle |
| Celeb-DF v2 | Respaldo cross-dataset | Formulario académico |

**Estado actual:** El acceso a FF++ ya fue aprobado y verificamos la descarga de una muestra (10 videos por categoría, c23) en un notebook de Kaggle, con extracción de frames y visualización real vs. falso.

**Preprocesamiento previsto:** Extraer 20–30 frames por video, detectar y recortar la cara (MTCNN, Zhang et al., 2016, o RetinaFace, Deng et al., 2020), redimensionar a 224×224 (299×299 si se respeta la entrada original de Xception). Partición por video, no por frame, siguiendo los splits oficiales de FF++ (Rössler et al., 2019), para evitar fuga de información entre entrenamiento y prueba.

**Licencia:** Los datos se usan solo con fines académicos, se guardan en un dataset privado de Kaggle compartido únicamente con los integrantes, y no se redistribuyen. Se citarán según lo exigen los autores (Rössler et al., 2019; Dolhansky et al., 2019; Y. Li et al., 2020).

# **5\. Plan por fases** {#5.-plan-por-fases}

## **Fase A: Disección instrumentada de Xception** {#fase-a:-disección-instrumentada-de-xception}

* Conteo de parámetros y FLOPs por bloque (entry / middle / exit flow; Chollet, 2017), comparando convolución separable vs. convolución estándar equivalente.  
* Mapa del campo receptivo por bloque.  
* Visualización de mapas de activación intermedios y Grad-CAM (Selvaraju et al., 2017\) sobre caras reales y falsas, siguiendo el uso de la explicabilidad en detección de deepfakes (Du et al., 2020; Xu et al., 2022).  
* Latencia por frame y memoria en GPU (Kaggle T4/P100).

## **Fase B: Intervención en el núcleo** {#fase-b:-intervención-en-el-núcleo}

**Cambio:** Reemplazar la pérdida binaria estándar (BCE) por focal loss, FL(pₜ) \= −α(1 − pₜ)^γ · log(pₜ). Todo lo demás se mantiene igual que en la línea base: arquitectura, datos, splits, semilla, épocas y optimizador. Como con γ \= 0 focal loss se reduce a BCE, la comparación queda controlada. Xu et al. (2022) ofrecen un antecedente de que modificar la función de pérdida puede influir en la generalización a deepfakes no vistos; el uso específico de focal loss con ese fin es una hipótesis propia de este trabajo.

**Hipótesis previa (escrita antes de correr):** Con BCE, el modelo aprende rápido los ejemplos fáciles, que suelen ser los que muestran los artefactos más obvios del generador de entrenamiento (Luo et al., 2021), y deja de aprender de ellos. Focal loss reduce el peso de esos ejemplos fáciles y concentra el gradiente en los difíciles (falsificaciones sutiles, frames comprimidos, caras de perfil). Esperamos que el AUC in-domain casi no cambie y que el AUC cross-manipulation mejore un poco, más frente a FaceSwap, que comparte artefactos de mezcla con Deepfakes (L. Li et al., 2020), que frente a NeuralTextures, cuyos artefactos son de otra naturaleza. También contemplamos que no mejore: si los ejemplos difíciles siguen teniendo la misma huella del generador, focal loss solo afinaría esa misma huella.

**Medición:** AUC in-domain y cross-manipulation con γ ∈ {0 (BCE), 1, 2, 5}, con los mismos splits. Se traza la curva γ vs. AUC por tipo de manipulación y se comparan los mapas Grad-CAM de BCE y del mejor γ.

**Ablación secundaria (si el tiempo alcanza):** Agregar al stem una rama de alta frecuencia (filtros SRM; Fridrich y Kodovsky, 2012; Zhou et al., 2018), en línea con el uso de ruido de alta frecuencia para generalizar (Luo et al., 2021), con su propia hipótesis previa.

## **Fase C: Estrés y honestidad** {#fase-c:-estrés-y-honestidad}

* Entrenar solo en Deepfakes (c23) y evaluar en Face2Face, FaceSwap y NeuralTextures, un protocolo comparable al de Haliassos et al. (2021).  
* Evaluar en c40, mayor compresión (Rössler et al., 2019), y en DFDC Preview (Dolhansky et al., 2019), otro dataset con otras condiciones de captura.  
* Diagnóstico de fallos con Grad-CAM y ejemplos mal clasificados, conectando con transfer learning y domain shift (Verdoliva, 2020; Brodarič et al., 2024), y reportando explícitamente lo que no funcione.

# **6\. Métricas, cómputo y riesgos** {#6.-métricas,-cómputo-y-riesgos}

**Métrica principal:** AUC-ROC a nivel de frame y de video (promedio de frames), como en Haliassos et al. (2021).

**Secundarias:** Accuracy, matriz de confusión, caída relativa de AUC (in-domain → fuera de dominio), latencia y FLOPs.

**Cómputo:** Kaggle Notebooks con GPU (cuota semanal de \~30 h por cuenta, tres cuentas). Datos en un dataset privado de Kaggle compartido entre los integrantes para no repetir descargas.

| Riesgo | Mitigación |
| :---- | :---- |
| Tiempo de GPU insuficiente | Subconjunto de videos (p. ej. 200–300 por clase) y pocos frames por video; recortes precalculados |
| No reproducir la línea base | Reportar la brecha con honestidad y documentar diferencias de preprocesamiento |
| Acceso o tamaño de DFDC | Usar Celeb-DF v2 como dataset alternativo de shift |
| La intervención no mejora | Es un resultado válido: se reporta y se analiza contra la hipótesis previa |

# **7\. Cronograma y reparto tentativo** {#7.-cronograma-y-reparto-tentativo}

| Fecha | Hito |
| :---- | :---- |
| 07 oct | E1 · Propuesta de tema (este documento) |
| 08–14 oct | Pipeline de datos (frames \+ recorte de caras) y línea base Xception con BCE in-domain |
| 15–21 oct | Teardown (Fase A) \+ primera corrida con focal loss \+ cuaderno de decisiones en GitHub → **E2 (21 oct)** |
| 22 oct – 4 nov | Barrido de γ (Fase B) y evaluación bajo shift (Fase C) |
| 5–11 nov | Demo reproducible, informe final (4–8 pág., plantilla WAIMLAP) → **E3 (11 nov)** |
| 18–20 nov | Póster en WAIMLAP → **E4** |

**Reparto inicial (rotativo; todos deben poder defender cada decisión):**

* Pipeline de datos, splits y dataset compartido en Kaggle.  
* Modelo base, entrenamiento con BCE y con focal loss.  
* Teardown (FLOPs, campo receptivo), Grad-CAM y evaluación bajo shift.

La asignación de cada frente a cada integrante se registrará en el cuaderno de decisiones.

# **8\. Uso responsable de datos y de IA** {#8.-uso-responsable-de-datos-y-de-ia}

**Datos:** Los videos contienen rostros de personas reales. Se usarán solo para investigación académica, sin redistribución, sin intentar identificar a las personas y sin generar nuevos deepfakes, conforme a los términos de uso de cada dataset (Rössler et al., 2019; Dolhansky et al., 2019; Y. Li et al., 2020). Reconocemos un posible sesgo demográfico en los datasets (distribución de edad, género y tono de piel), que se mencionará como limitación en el análisis de fallos. También reconocemos que los generadores de FF++ son anteriores a los basados en difusión (Brodarič et al., 2024), por lo que nuestras conclusiones no se extienden a ellos.

**Asistentes de IA:** Se usaron para explorar temas, redactar esta propuesta y preparar el notebook de descarga y exploración de datos. El cuaderno de decisiones indicará en cada paso dónde hubo asistencia de IA; todas las decisiones de diseño serán revisadas y defendidas por el equipo.

# **9\. Referencias** {#9.-referencias}

1. Amada, T., Kakizaki, K., Miyagawa, T., Ebihara, A. F., Shiohara, K., & Yamasaki, T. (2025). *Robust deepfake detection for electronic know your customer systems using registered images*. arXiv. [https\://arxiv.org/abs/2507.22601](https://arxiv.org/abs/2507.22601)  
2. Brodarič, M., Peer, P., & Štruc, V. (2024). Cross-dataset deepfake detection: Evaluating the generalization capabilities of modern DeepFake detectors. In *Proceedings of the 27th Computer Vision Winter Workshop (CVWW)*.  
3. Chollet, F. (2017). Xception: Deep learning with depthwise separable convolutions. In *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)* (pp. 1251–1258). [https\://doi.org/10.1109/CVPR.2017.195](https://doi.org/10.1109/CVPR.2017.195)  
4. Deng, J., Guo, J., Ververas, E., Kotsia, I., & Zafeiriou, S. (2020). RetinaFace: Single-shot multi-level face localisation in the wild. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)* (pp. 5203–5212).  
5. Dolhansky, B., Howes, R., Pflaum, B., Baram, N., & Canton Ferrer, C. (2019). *The Deepfake Detection Challenge (DFDC) preview dataset*. arXiv. [https\://arxiv.org/abs/1910.08854](https://arxiv.org/abs/1910.08854)  
6. Du, M., Pentyala, S., Li, Y., & Hu, X. (2020). Towards generalizable deepfake detection with locality-aware autoencoder. In *Proceedings of the 29th ACM International Conference on Information & Knowledge Management (CIKM)* (pp. 325–334). [https\://arxiv.org/abs/1909.05999](https://arxiv.org/abs/1909.05999)  
7. Fridrich, J., & Kodovsky, J. (2012). Rich models for steganalysis of digital images. *IEEE Transactions on Information Forensics and Security, 7*(3), 868–882. [https\://doi.org/10.1109/TIFS.2012.2190402](https://doi.org/10.1109/TIFS.2012.2190402)  
8. Haliassos, A., Vougioukas, K., Petridis, S., & Pantic, M. (2021). Lips don’t lie: A generalisable and robust approach to face forgery detection. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)* (pp. 5039–5049). [https\://arxiv.org/abs/2012.07657](https://arxiv.org/abs/2012.07657)  
9. Jeswani, U., & Priyadharshini, S. (2026). Exploring facial feature extraction and temporal dynamics for deepfake video detection: A CNN-RNN study on FaceForensics++. *ICITIIT*.  
10. Li, L., Bao, J., Zhang, T., Yang, H., Chen, D., Wen, F., & Guo, B. (2020). Face X-ray for more general face forgery detection. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)* (pp. 5001–5010). [https\://doi.org/10.1109/CVPR42600.2020.00505](https://doi.org/10.1109/CVPR42600.2020.00505)  
11. Li, Y., Yang, X., Sun, P., Qi, H., & Lyu, S. (2020). Celeb-DF: A large-scale challenging dataset for DeepFake forensics. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)* (pp. 3207–3216).  
12. Lin, T.-Y., Goyal, P., Girshick, R., He, K., & Dollár, P. (2017). Focal loss for dense object detection. In *Proceedings of the IEEE International Conference on Computer Vision (ICCV)* (pp. 2980–2988). [https\://doi.org/10.1109/ICCV.2017.324](https://doi.org/10.1109/ICCV.2017.324)  
13. Luo, Y., Zhang, Y., Yan, J., & Liu, W. (2021). Generalizing face forgery detection with high-frequency features. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)* (pp. 16317–16326). [https\://doi.org/10.1109/CVPR46437.2021.01605](https://doi.org/10.1109/CVPR46437.2021.01605)  
14. Rana, P., Kumar, L., Gupta, R., Dhiman, S., Bagga, V., & Gupta, B. (2026). Optimizing deepfake detection transformer on FaceForensics++: A hyperparameter grid search study. *ICBDML*.  
15. Rössler, A., Cozzolino, D., Verdoliva, L., Riess, C., Thies, J., & Nießner, M. (2019). FaceForensics++: Learning to detect manipulated facial images. In *Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)* (pp. 1–11). [https\://doi.org/10.1109/ICCV.2019.00009](https://doi.org/10.1109/ICCV.2019.00009)  
16. Selvaraju, R. R., Cogswell, M., Das, A., Vedantam, R., Parikh, D., & Batra, D. (2017). Grad-CAM: Visual explanations from deep networks via gradient-based localization. In *Proceedings of the IEEE International Conference on Computer Vision (ICCV)* (pp. 618–626). [https\://doi.org/10.1109/ICCV.2017.74](https://doi.org/10.1109/ICCV.2017.74)  
17. Tan, M., & Le, Q. (2019). EfficientNet: Rethinking model scaling for convolutional neural networks. In *Proceedings of the 36th International Conference on Machine Learning (ICML)* (pp. 6105–6114). PMLR.  
18. Verdoliva, L. (2020). Media forensics and DeepFakes: An overview. *IEEE Journal of Selected Topics in Signal Processing, 14*(5), 910–932. [https\://doi.org/10.1109/JSTSP.2020.3002101](https://doi.org/10.1109/JSTSP.2020.3002101)  
19. Xu, Y., Raja, K., & Pedersen, M. (2022). Supervised contrastive learning for generalizable and explainable DeepFakes detection. In *Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision Workshops (WACVW)*. [https\://openaccess.thecvf.com/content/WACV2022W/XAI4B/papers/Xu\_Supervised\_Contrastive\_Learning\_for\_Generalizable\_and\_Explainable\_DeepFakes\_Detection\_WACVW\_2022\_paper.pdf](https://openaccess.thecvf.com/content/WACV2022W/XAI4B/papers/Xu_Supervised_Contrastive_Learning_for_Generalizable_and_Explainable_DeepFakes_Detection_WACVW_2022_paper.pdf)  
20. Zhang, K., Zhang, Z., Li, Z., & Qiao, Y. (2016). Joint face detection and alignment using multitask cascaded convolutional networks. *IEEE Signal Processing Letters, 23*(10), 1499–1503.  
21. Zhou, P., Han, X., Morariu, V. I., & Davis, L. S. (2018). Learning rich features for image manipulation detection. In *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)* (pp. 1053–1061).