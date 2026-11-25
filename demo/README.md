# Demo en vivo: Grad-CAM de los modelos de la Fase B y de la Fase D

Interfaz local (Gradio) para mostrar qué mira el detector. Se sube una imagen y la app:

1. detecta la cara más grande con MTCNN y la recorta como en el entrenamiento (caja cuadrada 1.3×, 299×299);
2. pasa el recorte por los cuatro modelos de la semilla 42: **Fase B** (solo Deepfakes; BCE y Focal γ = 2) y **Fase D** (Deepfakes + SBI; R1 = SBI + BCE y R3 = SBI + Focal + JPEG, el modelo final);
3. muestra dos a la vez según la comparación elegida (*Fase B*, *Fase D* o *B vs. D*: BCE frente a R3). Cambiar de comparación no vuelve a correr los modelos.

Español e inglés con el selector 🌐 (los enlaces de Gradio de la barra siguen el idioma del navegador). Cada tarjeta muestra:

| Elemento | Qué indica |
|---|---|
| Decisión | **EVIDENCIA DE FALSO** si p(falso) ≥ 0.5; si no, **SIN EVIDENCIA**. No se usa «real»: el modelo solo reconoce las manipulaciones que aprendió, así que no detectar no prueba que la imagen sea auténtica |
| «En el test detecta» | % de cada manipulación marcada como falsa con umbral 0.5 en el test de FF++ (y % de reales bien clasificados). `preds_*.csv.gz` de los notebooks 04 (Fase B) y 07 (Fase D), todos los frames |
| Barra de regiones | masa del Grad-CAM en el interior (naranja), borde (morado) y fuera (azul) |
| Patrón | interior si masa interior ≥ 0.60; fuera si masa fuera ≥ 0.30. ✓ si coincide con lo medido en el test (detecta ↔ interior; no detecta ↔ fuera), ⚠ si no |

| Modelo (s42) | DF | F2F | FS | NT | Reales bien |
|---|---|---|---|---|---|
| BCE (Fase B) | 97 % | 12 % | 1 % | 21 % | 95 % |
| Focal γ = 2 (Fase B) | 97 % | 7 % | 0 % | 17 % | 96 % |
| R1 · SBI + BCE (Fase D) | 96 % | 17 % | 3 % | 19 % | 97 % |
| R3 · SBI + Focal + JPEG (Fase D) | 99 % | 58 % | 25 % | 51 % | 87 % |

Por eso, con una foto editada, filtrada o hecha con otro generador lo esperable es «sin evidencia», sobre todo en la Fase B. Las secciones «Cómo leer los resultados» y «Resultados del proyecto y límites» lo resumen dentro de la app.

**Verificación (24/11):** con recortes del test de FF++ y la opción «La imagen ya es un recorte de la cara», la demo reproduce las probabilidades de `preds_*.csv.gz` de Kaggle en los cuatro modelos (notebooks 04 y 07; 30 frames por modelo, diferencia máxima 0.007, ninguna decisión distinta).

## Requisitos

- Pesos (~80 MB cada uno) en `results/checkpoints/`, que Git ignora. **No se suben al repositorio:** son un derivado de FF++.
  - Fase B (obligatorios): `bce_s42.pt` y `focal_g2_s42.pt`, del output del notebook 04 v3 (carpeta `checkpoints/`).
  - Fase D (opcionales): `r1_sbi_bce_s42.pt` y `r3_sbi_focal_g2_aug_s42.pt`, del output del notebook 07. Sin ellos la app arranca solo con la comparación de la Fase B y avisa qué falta.
- Python con PyTorch en CPU (probado con Python 3.14, torch 2.15 CPU, timm 1.0, Gradio 6). Con los dos modelos tarda ~0.5 s por imagen en CPU.

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1   ·   Linux/macOS: source .venv/bin/activate
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r demo/requirements.txt
python -m pip install --no-deps facenet-pytorch
```

`facenet-pytorch` se instala sin dependencias porque fija versiones antiguas de torch, numpy y Pillow; funciona con las actuales (pip avisa del conflicto, se puede ignorar).

## Uso

```bash
python demo/app.py                       # http://127.0.0.1:7860
python demo/app.py --ckpt-dir RUTA       # si los pesos están en otra carpeta
```

Botones *Analizar* y *Limpiar* bajo la imagen. En *Opciones*: «La imagen ya es un recorte de la cara» salta la detección; «Dibujar los límites de las regiones» muestra los recuadros del interior (lado 0.5 del recorte) y de la cara (lado 1/1.3).

**Respaldo en Kaggle:** notebook privado con el output del 04 como input, `pip install gradio` y `python demo/app.py --share` desde el repo clonado. La app busca los pesos también en `/kaggle/input`. El enlace de `--share` es público mientras dure la sesión: cerrarlo al terminar.

## Guion sugerido (~5 min)

1. **Foto de un integrante** (con su consentimiento), en *Fase B*. Lo esperado: «sin evidencia» y masa fuera de la cara. Probarla antes: una foto de cámara está fuera de la distribución de FF++ y puede dar falsa alarma, sobre todo con R3.
2. **Deepfake:** los dos modelos lo detectan, con la masa en el interior de la cara.
3. **FaceSwap, en *B vs. D*:** BCE no encuentra evidencia; R3 tiene más probabilidad de detectarlo (25 % en el test, frente al 1 %), y cuando lo hace la masa pasa al interior.
4. **Cierre:** Focal cambia poco (Fase B); lo que generalizó fue cambiar los datos (Fase D: FaceSwap 0.26 → 0.76 de AUC), con el costo de más falsas alarmas.

## Licencia y privacidad

- La app no guarda imágenes ni resultados. Gradio guarda temporalmente lo que se sube en su caché; la app la borra cada 10 minutos y al cerrarse.
- Las caras de FF++ solo se muestran en vivo, desde la máquina del grupo: no se graban, no se capturan y no se publican.
- No se agregan imágenes de ejemplo al repositorio.
