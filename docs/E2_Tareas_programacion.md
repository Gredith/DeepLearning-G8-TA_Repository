# E2 · Tareas de programación

**Proyecto:** Robustez cross-manipulation de Xception con focal loss (Equipo 8)
**Entrega:** miércoles 21 de octubre de 2026
**Alcance de este archivo:** solo lo que hay que programar y correr. La redacción del cuaderno de decisiones y la actualización del texto de la propuesta van aparte.

> **Nota sobre el código:** los módulos de T1–T7 y la pérdida/pruebas de T3 están implementados. Los notebooks E2 y `src/kaggle_e2/` cubren el protocolo por subconjunto; el pipeline CLI completo está en `src/data_pipeline/`, `src/train.py`, `src/evaluate*.py` y `src/teardown/`. Consulta `docs/DECISIONS.md` para distinguir los resultados sintéticos/CPU ya generados de las corridas pendientes en GPU con FF++. T8/T9 y el registro de versiones del entorno siguen pendientes.

> **Cómo leer las casillas:** una casilla marcada indica que el código o la documentación correspondiente existe. Las casillas de ejecución y los criterios de aceptación permanecen sin marcar hasta que existan resultados generados con FF++ y un runtime compatible.

### Seguimiento del repositorio

- [x] Crear la estructura inicial de carpetas y módulos descrita en este E2.
- [x] Actualizar README con el flujo previsto, preparación del entorno y estado real de implementación.
- [x] Añadir `.gitignore` para entornos Python, datos locales y artefactos generados.
- [x] Añadir `configs/base.yaml`, `docs/DECISIONS.md` y dependencias iniciales.
- [x] Implementar la clase base `BinaryFocalLoss` en `src/models/losses.py`.
- [x] Reorganizar documentación, código y dependencias para mantener limpio el root.
- [x] Implementar T1: descarga/validación de splits por pares, manifiestos, extracción uniforme, recorte MTCNN, dataset normalizado y conteos.
- [x] Descargar los JSON oficiales y guardar listas normalizadas de IDs fuente con hashes en `results/splits/` (720/140/140 IDs únicos; sin solapamiento entre particiones).
- [ ] Ejecutar T1 completo en Kaggle con FF++ autorizado; revisar el grid de 16 caras por clase y guardar manifiestos/resultados.
- [x] Implementar T2: entrenamiento reproducible, BCE/focal intercambiable, AUC por video/frame, mejor checkpoint, historial y curva.
- [x] Añadir pruebas T3 para γ=0, ejemplos fáciles/difíciles, logits extremos, gradientes y argumentos.
- [ ] Ejecutar pruebas T3 en un entorno con PyTorch y correr baseline BCE + focal γ=2.
- [x] Implementar T4: parámetros/FLOPs por flujo y ahorro separable vs. convolución estándar.
- [x] Implementar T7: campo receptivo con rutas residuales y medición de latencia/memoria.
- [ ] Ejecutar T4/T7 con `timm`, `fvcore` y la GPU destino; guardar sus tablas/figuras.
- [x] Implementar T5: selección fija de ejemplos y Grad-CAM BCE/focal en la capa `conv4`.
- [x] Implementar T6: evaluación in-domain/cross-manipulation y caída relativa de AUC.
- [ ] Ejecutar T5/T6 con los checkpoints y registrar sus figuras/tablas.
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
├── docs/
│   ├── E1 _Propuesta de tema_Grupo_8.md
│   ├── E2_Tareas_programacion.md
│   ├── DECISIONS.md
│   └── PROJECT_STRUCTURE.md
├── environment/
│   └── requirements.txt
├── configs/
│   └── base.yaml              # semilla, lr, batch, épocas, rutas, γ, α
├── src/
│   ├── train.py
│   ├── evaluate.py
│   ├── evaluate_cross.py
│   ├── data_pipeline/
│   │   ├── build_splits.py
│   │   ├── extract_frames.py
│   │   ├── crop_faces.py
│   │   └── dataset.py
│   ├── models/                 # Xception y pérdidas
│   └── teardown/               # FLOPs, campo receptivo, latencia, Grad-CAM
├── notebooks/
│   └── kaggle/                 # exploración/experimentos en Kaggle
├── tests/                      # pruebas unitarias y de sanidad
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
- [x] Descargar los JSON oficiales de splits FF++ (360/70/70 pares que contienen 720/140/140 IDs fuente únicos).
- [x] Guardar en `results/splits/` los pares originales, las listas normalizadas de IDs y hashes de origen.
- [x] **Regla:** la partición es por video, nunca por frame; el validador rechaza IDs compartidos.

### 2.2 Extracción de frames
- [x] Implementar extracción de 20–30 frames equiespaciados por video (OpenCV; valor configurado: 25).
- [x] Implementar el inventario/procesamiento de `original` y `Deepfakes` para entrenamiento; `Face2Face`, `FaceSwap`, `NeuralTextures` para evaluación.

### 2.3 Recorte de caras
- [x] Elegir MTCNN (`facenet-pytorch`) y registrarlo en `docs/DECISIONS.md`.
- [x] Implementar detector facial MTCNN para procesar los frames de FF++.
- [x] Implementar margen alrededor de la caja (1.3×) y recorte cuadrado.
- [x] Implementar redimensionado a 299×299 y guardado de recortes precalculados en disco.
- [x] Implementar registro de frames sin cara detectada y descarte consistente.

### 2.4 Dataset de PyTorch
- [x] `Dataset` que devuelva `(imagen, etiqueta, video_id, manipulacion)`. El `video_id` es necesario para el AUC a nivel de video.
- [x] Etiqueta: real = 0, falso = 1.
- [x] Normalización configurable con media/desviación `[0.5, 0.5, 0.5]`, según `default_cfg` de `legacy_xception` en `timm` (D-02). Esta decisión específica de la arquitectura prevalece sobre el valor inicial de ImageNet indicado en el borrador de E2.
- [x] Aumentación mínima en entrenamiento (flip horizontal). Cualquier otra, a `docs/DECISIONS.md`.

### 2.5 Pruebas de sanidad
- [x] **Test de fuga:** la intersección de IDs fuente entre train, val y test debe ser vacía (`assert`); validado con los JSON oficiales.
- [x] Implementar conteo de muestras y balance de clases por partición.
- [x] Implementar generación de una grilla con 16 recortes reales y 16 falsos; la revisión visual queda pendiente de correr con FF++.

**Criterio de aceptación:** `src/data_pipeline/dataset.py` entrega batches de forma `[B, 3, 299, 299]`, el test de fuga pasa en los splits oficiales y existen tablas con conteos por partición y clase. La verificación con datos reales queda pendiente de ejecutar en Kaggle.

---

## 3. T2 · Línea base Xception con BCE (P0)

### 3.1 Modelo
```python
# src/models/xception.py
import timm, torch.nn as nn

def build_xception(pretrained=True):
    # En timm reciente, el Xception original puede llamarse 'legacy_xception'.
    # Verificar con timm.list_models('*xception*').
    model = timm.create_model("legacy_xception", pretrained=pretrained, num_classes=1)
    return model  # salida: un logit por imagen
```

### 3.2 Entrenamiento (`src/train.py`)
- [x] Fijar semillas (`random`, `numpy`, `torch`, `torch.cuda`) y `torch.backends.cudnn.deterministic = True`.
- [x] Un solo logit con `BCEWithLogitsLoss` como pérdida base.
- [x] Optimizador, lr, batch size y épocas definidos en `configs/base.yaml` y compartidos entre BCE y focal.
- [x] Guardar el mejor checkpoint por AUC de validación.
- [x] Registrar por época: pérdida de train y val, AUC de val.
- [x] Mixed precision configurable para aprovechar GPU.
- [x] Guardar la config completa y el hash del commit junto al checkpoint.

### 3.3 Evaluación in-domain (`src/evaluate.py`)
- [x] Implementar AUC-ROC a nivel de frame.
- [x] Implementar AUC a nivel de video promediando probabilidades por `video_id`.
- [x] Implementar accuracy y matriz de confusión.
- [x] Implementar guardado de métricas JSON/CSV en `results/metrics/`.

**Criterio de aceptación:** existe un checkpoint, una curva de entrenamiento y un JSON con AUC in-domain por frame y por video. Se compara con el valor reportado por Rössler et al. (2019) y la brecha, si la hay, se anota. **No es necesario cerrarla para E2.**

---

## 4. T3 · Focal loss (P0)

### 4.1 Implementación
```python
# src/models/losses.py
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

> **Punto de diseño a decidir y registrar:** la fórmula de la propuesta incluye α. Con α fijo (p. ej. 0.25) y γ=0, la pérdida **no** es igual a BCE, sino BCE escalada por clase. Para que el control "γ=0 ⇒ BCE" sea exacto, usen `alpha=None` (o compárenlo contra BCE ponderada). Decidan una opción, úsenla en todas las corridas y regístrenla en `docs/DECISIONS.md`.

### 4.2 Tests obligatorios (`tests/test_losses.py`)
- [x] Test: **γ=0, α=None** coincide con `BCEWithLogitsLoss` (`torch.allclose`, tolerancia 1e-6) sobre logits aleatorios.
- [x] Test: con γ>0, un ejemplo fácil (p_t≈0.99) recibe mucho menos peso que uno difícil (p_t≈0.5).
- [x] Test: no produce `NaN` con logits extremos (±50).
- [x] Test: gradientes finitos. (Pendiente de ejecutar en runtime con PyTorch.)

### 4.3 Corridas
- [ ] **γ=0** con el código de focal: debe dar un resultado muy cercano a la línea base BCE (mismo `seed`).
- [ ] **γ=2** (corrida principal de E2).
- [ ] **γ=1** si alcanza el tiempo (T8).
- [ ] Cambiar **solo** la pérdida. Mismos datos, splits, semilla, épocas y optimizador que T2.

**Criterio de aceptación:** los tests pasan, existe al menos una corrida con γ=2 y su JSON de métricas, y queda escrito qué diferencia hubo respecto de BCE.

---

## 5. T4 · Teardown: parámetros y FLOPs por bloque (P0)

### 5.1 Parámetros y FLOPs por bloque
- [x] Implementar agrupación en **entry flow** (conv1, conv2, block1–3), **middle flow** (block4–11) y **exit flow** (block12, conv3, conv4); validar nombres al cargar el modelo.
- [x] Implementar conteo de parámetros por bloque.
- [x] Implementar conteo de FLOPs por bloque con `fvcore`.
- [x] Implementar tabla: bloque · parámetros · % del total · FLOPs · % del total · forma de salida.

### 5.2 Separable vs. convolución estándar
- [x] Implementar cálculo comparativo separable vs. convolución estándar con los mismos canales y kernel.
- [x] Implementar reporte del factor de ahorro por bloque.
- [x] Implementar verificación analítica: separable ≈ `k²·C_in + C_in·C_out` frente a `k²·C_in·C_out`.

**Criterio de aceptación:** una tabla en `results/metrics/teardown_flops.csv` y una figura de barras por bloque. Cada resultado va con una lectura de una o dos frases (¿dónde se concentra el cómputo? ¿dónde los parámetros?).

---

## 6. T5 · Grad-CAM (P1)

- [x] Implementar Grad-CAM a mano (Selvaraju et al., 2017).
- [x] Elegir y validar la capa objetivo `conv4` (última convolución antes del pooling) al cargar el modelo.
- [x] Implementar selección fija y emparejada de ejemplos para los modelos BCE y focal (categorías disponibles según predicciones del modelo BCE).
  - [x] 8 reales bien clasificadas
  - [x] 8 falsas bien clasificadas
  - [x] 8 falsas mal clasificadas (falsos negativos)
  - [x] 8 reales mal clasificadas (falsos positivos)
- [x] Implementar grilla: imagen · Grad-CAM BCE · Grad-CAM focal.
- [x] Implementar guardado de IDs de imágenes seleccionadas para reproducir la figura.
- [ ] Opcional: repetir sobre imágenes de FaceSwap y NeuralTextures (para T6).

**Criterio de aceptación:** una figura en `results/figures/` con la grilla y una lectura escrita de qué regiones usa cada modelo. Si no hay diferencia visible, se dice.

---

## 7. T6 · Evaluación cross-manipulation preliminar (P1)

- [x] Implementar carga de checkpoints entrenados con Deepfakes.
- [x] Implementar evaluación en Face2Face, FaceSwap y NeuralTextures con caras `original` del mismo split.
- [x] Implementar reporte de AUC por frame y por video para BCE y focal.
- [x] Implementar cálculo de la caída relativa de AUC: `(AUC_in-domain − AUC_cross) / AUC_in-domain`.
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
- [x] Implementar cálculo analítico del campo receptivo, stride acumulado y *jump* tras cada capa/bloque de Xception.
- [ ] Recordar que las conexiones residuales no aumentan el campo receptivo máximo, pero sí aportan rutas más cortas. Anotar la lectura.
- [x] Implementar tabla: bloque · stride acumulado · campo receptivo · resolución del mapa de salida.
- [x] Implementar verificación de formas espaciales con el modelo cargado.

### 8.2 Latencia y memoria
- [x] Implementar calentamiento configurable (20 por defecto).
- [x] Implementar sincronización CUDA antes y después de las mediciones.
- [x] Implementar medición con batch=1 y batch=32; media y desviación sobre ≥100 iteraciones por defecto.
- [x] Implementar medición de memoria pico en inferencia y entrenamiento.
- [x] Implementar registro del dispositivo/GPU utilizado.

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

- [ ] `environment/requirements.txt` con versiones fijadas (`torch`, `timm`, `scikit-learn`, `fvcore` o equivalente, `pytorch-grad-cam`, `facenet-pytorch`).
- [x] `README.md` con instrucciones para obtener/preprocesar datos, entrenar, evaluar y lanzar los análisis (las instrucciones deben validarse al ejecutar el flujo en Kaggle).
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
