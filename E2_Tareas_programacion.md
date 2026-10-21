# E2 · Tareas de programación

**Proyecto:** Robustez cross-manipulation de Xception con focal loss (Equipo 8)
**Entrega:** miércoles 21 de octubre de 2026
**Alcance de este archivo:** solo lo que hay que programar y correr. La redacción del cuaderno de decisiones y la actualización del texto de la propuesta van aparte.

> **Nota sobre el código:** los fragmentos de este documento son una guía. La estructura inicial ya está creada, pero los puntos de entrada del pipeline, entrenamiento y análisis siguen pendientes de implementación/validación. Los nombres de capas de `timm` cambian entre versiones, así que verifíquenlos con `print(model)` antes de usarlos.

### Seguimiento del repositorio

- [x] Crear la estructura inicial de carpetas y módulos descrita en este E2.
- [x] Actualizar README con el flujo previsto, preparación del entorno y estado real de implementación.
- [x] Añadir `.gitignore` para entornos Python, datos locales y artefactos generados.
- [x] Añadir `configs/base.yaml`, `DECISIONS.md` y dependencias iniciales.
- [x] Implementar la clase base `BinaryFocalLoss` en `models/losses.py`.
- [ ] Fijar versiones de dependencias después de validar el entorno de ejecución.
- [ ] Completar y ejecutar los criterios de aceptación T1–T9; no se considera ninguna tarea experimental terminada por crear su esqueleto.

---

## 0. Prioridades

Si el tiempo es poco, este es el orden de corte. **P0 es lo mínimo defendible; P2 se sacrifica primero.**

| Prioridad | Tarea | Depende de |
|---|---|---|
| **P0** | T1 Pipeline de datos | — |
| **P0** | T2 Línea base BCE | T1 |
| **P0** | T3 Focal loss validada (γ=0 ≈ BCE) y una corrida con γ=2 | T1, T2 |
| **P0** | T4 Teardown: parámetros/FLOPs por bloque | — (no necesita datos) |
| **P1** | T5 Grad-CAM BCE vs. focal | T2, T3 |
| **P1** | T6 Evaluación cross-manipulation preliminar | T2, T3 |
| **P1** | T7 Campo receptivo, latencia y memoria | — |
| **P2** | T8 Corrida con γ=1 | T3 |
| **P2** | T9 Evaluación c40 y DFDC | T2, T3 |

**Lo que puede correr en paralelo desde el minuto cero:** T4 y T7 no necesitan datos ni entrenamiento. Mientras T2 entrena, avancen T4 y T7.

---

## 1. Estructura del repositorio

```
.
├── README.md
├── requirements.txt
├── DECISIONS.md
├── configs/
│   └── base.yaml              # semilla, lr, batch, épocas, rutas, γ, α
├── data_pipeline/
│   ├── extract_frames.py
│   ├── crop_faces.py
│   ├── build_splits.py
│   └── dataset.py
├── models/
│   ├── xception.py
│   └── losses.py
├── train.py
├── evaluate.py
├── teardown/
│   ├── flops_params.py
│   ├── receptive_field.py
│   ├── latency_memory.py
│   └── gradcam.py
├── notebooks/
│   └── 00_resumen_resultados.ipynb
└── results/
    ├── checkpoints/
    ├── metrics/               # JSON/CSV por corrida
    ├── figures/
    └── splits/
```

**Criterio de aceptación:** un compañero clona el repo, sigue el README y reproduce una corrida corta sin preguntar nada.

---

## 2. T1 · Pipeline de datos (P0)

### 2.1 Splits oficiales por video
- [ ] Descargar o localizar los JSON oficiales de splits de FF++ (train/val/test, 720/140/140 videos por clase).
- [ ] Guardar en `results/splits/` la lista de IDs de video por partición.
- [ ] **Regla:** la partición es por video, nunca por frame.

### 2.2 Extracción de frames
- [ ] Extraer 20–30 frames equiespaciados por video (OpenCV).
- [ ] Procesar: `original` (real) y `Deepfakes` en c23 para entrenamiento; `Face2Face`, `FaceSwap`, `NeuralTextures` solo para evaluación.

### 2.3 Recorte de caras
- [ ] Detectar la cara con MTCNN (`facenet-pytorch`) o RetinaFace; elegir una y registrarla en `DECISIONS.md`.
- [ ] Margen alrededor de la caja (p. ej. 1.3×) y recorte cuadrado.
- [ ] Redimensionar a la entrada de Xception (299×299) y **guardar los recortes precalculados** en disco. No recalcular en cada época.
- [ ] Registrar los frames sin cara detectada y descartarlos de forma consistente.

### 2.4 Dataset de PyTorch
- [ ] `Dataset` que devuelva `(imagen, etiqueta, video_id, manipulacion)`. El `video_id` es necesario para el AUC a nivel de video.
- [ ] Etiqueta: real = 0, falso = 1.
- [ ] Normalización con media/desviación de ImageNet.
- [ ] Aumentación mínima en entrenamiento (flip horizontal). Cualquier otra, a `DECISIONS.md`.

### 2.5 Pruebas de sanidad
- [ ] **Test de fuga:** la intersección de `video_id` entre train, val y test debe ser vacía (`assert`).
- [ ] Contar muestras y balance de clases por partición.
- [ ] Guardar una grilla de 16 recortes reales y 16 falsos y mirarla a ojo.

**Criterio de aceptación:** `dataset.py` entrega batches de forma `[B, 3, 299, 299]`, el test de fuga pasa y existe una tabla con conteos por partición y clase.

---

## 3. T2 · Línea base Xception con BCE (P0)

### 3.1 Modelo
```python
# models/xception.py
import timm, torch.nn as nn

def build_xception(pretrained=True):
    # En timm reciente, el Xception original puede llamarse 'legacy_xception'.
    # Verificar con timm.list_models('*xception*').
    model = timm.create_model("legacy_xception", pretrained=pretrained, num_classes=1)
    return model  # salida: un logit por imagen
```

### 3.2 Entrenamiento (`train.py`)
- [ ] Fijar semillas (`random`, `numpy`, `torch`, `torch.cuda`) y `torch.backends.cudnn.deterministic = True`.
- [ ] Un solo logit con `BCEWithLogitsLoss` como pérdida base.
- [ ] Optimizador, lr, batch size y épocas definidos en `configs/base.yaml` y **no modificables entre BCE y focal**.
- [ ] Guardar el mejor checkpoint por AUC de validación.
- [ ] Registrar por época: pérdida de train y val, AUC de val.
- [ ] Mixed precision (`torch.cuda.amp`) para ahorrar tiempo en T4/P100.
- [ ] Guardar la config completa y el hash del commit junto al checkpoint.

### 3.3 Evaluación in-domain (`evaluate.py`)
- [ ] AUC-ROC a nivel de frame (`sklearn.metrics.roc_auc_score`).
- [ ] AUC a nivel de video: promedio de las probabilidades de los frames de cada `video_id`.
- [ ] Accuracy, matriz de confusión.
- [ ] Guardar todo en `results/metrics/<run_name>.json`.

**Criterio de aceptación:** existe un checkpoint, una curva de entrenamiento y un JSON con AUC in-domain por frame y por video. Se compara con el valor reportado por Rössler et al. (2019) y la brecha, si la hay, se anota. **No es necesario cerrarla para E2.**

---

## 4. T3 · Focal loss (P0)

### 4.1 Implementación
```python
# models/losses.py
import torch, torch.nn as nn, torch.nn.functional as F

class BinaryFocalLoss(nn.Module):
    """FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)   (Lin et al., 2017)
    alpha=None -> sin ponderación por clase (necesario para que gamma=0 == BCE)."""
    def __init__(self, gamma=2.0, alpha=None, reduction="mean"):
        super().__init__()
        self.gamma, self.alpha, self.reduction = gamma, alpha, reduction

    def forward(self, logits, targets):
        logits = logits.view(-1)
        targets = targets.view(-1).float()
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p = torch.sigmoid(logits)
        p_t = p * targets + (1 - p) * (1 - targets)
        loss = (1 - p_t) ** self.gamma * bce
        if self.alpha is not None:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            loss = alpha_t * loss
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss
```

> **Punto de diseño a decidir y registrar:** la fórmula de la propuesta incluye α. Con α fijo (p. ej. 0.25) y γ=0, la pérdida **no** es igual a BCE, sino BCE escalada por clase. Para que el control "γ=0 ⇒ BCE" sea exacto, usen `alpha=None` (o compárenlo contra BCE ponderada). Decidan una opción, úsenla en todas las corridas y regístrenla en `DECISIONS.md`.

### 4.2 Tests obligatorios (`tests/test_losses.py`)
- [ ] **γ=0, α=None** coincide con `BCEWithLogitsLoss` (`torch.allclose`, tolerancia 1e-6) sobre logits aleatorios.
- [ ] Con γ>0, la pérdida de un ejemplo fácil (p_t≈0.99) es mucho menor que la de uno difícil (p_t≈0.5).
- [ ] No produce `NaN` con logits extremos (±50).
- [ ] Gradientes finitos.

### 4.3 Corridas
- [ ] **γ=0** con el código de focal: debe dar un resultado muy cercano a la línea base BCE (mismo `seed`).
- [ ] **γ=2** (corrida principal de E2).
- [ ] **γ=1** si alcanza el tiempo (T8).
- [ ] Cambiar **solo** la pérdida. Mismos datos, splits, semilla, épocas y optimizador que T2.

**Criterio de aceptación:** los tests pasan, existe al menos una corrida con γ=2 y su JSON de métricas, y queda escrito qué diferencia hubo respecto de BCE.

---

## 5. T4 · Teardown: parámetros y FLOPs por bloque (P0)

### 5.1 Parámetros y FLOPs por bloque
- [ ] Agrupar las capas en **entry flow** (conv1, conv2, block1–3), **middle flow** (block4–11) y **exit flow** (block12, conv3, conv4). Verificar los nombres con `print(model)`.
- [ ] Contar parámetros por bloque con `sum(p.numel() for p in block.parameters())`.
- [ ] Contar FLOPs por bloque con `fvcore.nn.FlopCountAnalysis`, `ptflops` o `thop` (elegir una y registrarla).
- [ ] Tabla: bloque · parámetros · % del total · FLOPs · % del total · forma de salida.

### 5.2 Separable vs. convolución estándar
- [ ] Para cada convolución separable, calcular los parámetros y FLOPs de una convolución estándar con el mismo `C_in`, `C_out` y kernel.
- [ ] Reportar el factor de ahorro por bloque.
- [ ] Verificación analítica: separable ≈ `k²·C_in + C_in·C_out` frente a `k²·C_in·C_out`.

**Criterio de aceptación:** una tabla en `results/metrics/teardown_flops.csv` y una figura de barras por bloque. Cada resultado va con una lectura de una o dos frases (¿dónde se concentra el cómputo? ¿dónde los parámetros?).

---

## 6. T5 · Grad-CAM (P1)

- [ ] Instalar `pytorch-grad-cam` o implementar Grad-CAM a mano (Selvaraju et al., 2017).
- [ ] Elegir la capa objetivo (última convolución antes del pooling). Verificar el nombre con `print(model)` y registrar la elección.
- [ ] Con el modelo **BCE** y el modelo **focal (γ=2)**, generar mapas para el **mismo conjunto fijo** de imágenes:
  - [ ] 8 reales bien clasificadas
  - [ ] 8 falsas bien clasificadas
  - [ ] 8 falsas mal clasificadas (falsos negativos)
  - [ ] 8 reales mal clasificadas (falsos positivos)
- [ ] Mostrar en una grilla: imagen · Grad-CAM BCE · Grad-CAM focal.
- [ ] Guardar los IDs de las imágenes elegidas para poder reproducir la figura.
- [ ] Opcional: repetir sobre imágenes de FaceSwap y NeuralTextures (para T6).

**Criterio de aceptación:** una figura en `results/figures/` con la grilla y una lectura escrita de qué regiones usa cada modelo. Si no hay diferencia visible, se dice.

---

## 7. T6 · Evaluación cross-manipulation preliminar (P1)

- [ ] Cargar el checkpoint entrenado solo con Deepfakes.
- [ ] Evaluar sobre los frames **de test** de Face2Face, FaceSwap y NeuralTextures. Para el lado real, usar las caras `original` **del mismo split de test**.
- [ ] Reportar AUC por frame y por video para BCE y para focal.
- [ ] Calcular la **caída relativa de AUC**: `(AUC_in-domain − AUC_cross) / AUC_in-domain`.
- [ ] Tabla resumen:

| Modelo | γ | In-domain (Deepfakes) | Face2Face | FaceSwap | NeuralTextures |
|---|---|---|---|---|---|
| BCE | – | | | | |
| Focal | 0 | | | | |
| Focal | 2 | | | | |

- [ ] Si hay una sola semilla, anótenlo en la tabla. Con diferencias pequeñas **no se pueden sacar conclusiones**; repetir con 2–3 semillas es lo ideal si el tiempo lo permite.

**Criterio de aceptación:** la tabla está completa para al menos BCE y γ=2, y se contrasta con la hipótesis previa (mejora, no mejora o empeora).

---

## 8. T7 · Campo receptivo, latencia y memoria (P1)

### 8.1 Campo receptivo por bloque
- [ ] Calcular analíticamente el campo receptivo, el stride acumulado y el *jump* tras cada capa/bloque, con kernels, strides y paddings de Xception.
- [ ] Recordar que las conexiones residuales no aumentan el campo receptivo máximo, pero sí aportan rutas más cortas. Anotar la lectura.
- [ ] Tabla: bloque · stride acumulado · campo receptivo · resolución del mapa de salida.
- [ ] Verificación empírica opcional: gradiente de una activación central respecto de la entrada.

### 8.2 Latencia y memoria
- [ ] Calentamiento de 10–20 iteraciones antes de medir.
- [ ] Usar `torch.cuda.synchronize()` antes y después de cada medición.
- [ ] Medir con batch=1 y batch=32; reportar media y desviación sobre ≥100 iteraciones.
- [ ] Memoria con `torch.cuda.max_memory_allocated()`, en inferencia y en entrenamiento.
- [ ] Registrar la GPU usada (T4 o P100), porque los números no son comparables entre ellas.

**Criterio de aceptación:** dos tablas en `results/metrics/` y la GPU anotada.

---

## 9. T8 y T9 · Opcionales (P2)

### T8 · γ=1
- [ ] Misma configuración que γ=2, solo cambia γ. Agregar la fila a la tabla de T6.

### T9 · c40 y DFDC
- [ ] Extraer frames y recortar caras de FF++ c40 con el mismo pipeline.
- [ ] Evaluar BCE y focal en c40.
- [ ] DFDC Preview: si el acceso o el tamaño es un problema, usar el subconjunto mínimo y anotar la limitación. Plan alternativo: Celeb-DF v2.

---

## 10. Reproducibilidad (revisar antes de entregar)

- [ ] `requirements.txt` con versiones fijadas (`torch`, `timm`, `scikit-learn`, `fvcore` o equivalente, `pytorch-grad-cam`, `facenet-pytorch`).
- [ ] `README.md` con: cómo obtener los datos, cómo preprocesar, cómo entrenar, cómo evaluar y cómo reproducir cada tabla y figura.
- [ ] Semilla y `commit hash` guardados con cada resultado.
- [ ] Ningún dato de FF++ dentro del repositorio (licencia). Solo código, splits y resultados.
- [ ] Los notebooks corren de arriba abajo sin errores (**Restart & Run All**).
- [ ] Las rutas no dependen de la máquina de una persona (usar `configs/base.yaml`).
- [ ] El repositorio es accesible para el profesor.

---

## 11. Reparto sugerido

| Persona | Tareas | Entregable |
|---|---|---|
| **A · Datos y base** | T1, T2, repo y README | Pipeline, checkpoint BCE, splits |
| **B · Intervención** | T3, T8, T6 | Focal loss con tests, corridas, tabla cross |
| **C · Teardown** | T4, T7, T5 | Tablas de FLOPs, campo receptivo, latencia, Grad-CAM |

Todos deben poder defender cualquier parte. Antes de entregar, cada persona explica el código de otra.

---

## 12. Checklist final de E2

- [ ] Pipeline funcionando, con test de fuga pasando
- [ ] Checkpoint BCE y AUC in-domain
- [ ] Focal loss implementada, con test γ=0 ≈ BCE
- [ ] Al menos una corrida focal (γ=2) con métricas
- [ ] Tabla de parámetros y FLOPs por bloque, con comparación separable vs. estándar
- [ ] Tabla de campo receptivo
- [ ] Grad-CAM BCE vs. focal en el mismo conjunto de imágenes
- [ ] Tabla cross-manipulation preliminar
- [ ] Latencia y memoria
- [ ] Repo reproducible y accesible
