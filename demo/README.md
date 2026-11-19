# Demo en vivo: Grad-CAM con BCE vs. Focal Loss (γ = 2)

Interfaz local (Gradio) para mostrar qué mira el detector. Se sube una imagen y la app:

1. detecta la cara más grande con MTCNN y la recorta como en el entrenamiento (caja cuadrada 1.3×, 299×299);
2. pasa el recorte por los dos modelos de la semilla 42 (BCE y Focal γ = 2, entrenados solo con Deepfakes de FF++ c23);
3. muestra, para cada uno, la probabilidad de «falso», la decisión (umbral 0.5), el Grad-CAM de `block12` y la masa del mapa por región (interior / borde / fuera y área que concentra el 50 %), la misma medición del notebook 06.

La interfaz está en **español e inglés**: selector 🌐 arriba a la derecha; el cambio es inmediato, no recarga las imágenes y también traduce los resultados ya calculados. En la misma barra, a la izquierda, quedan los enlaces de Gradio (*Usar vía API*, *Construido con Gradio*, *Configuración*); estos siguen el idioma del navegador, no el selector. Debajo de las imágenes, una tarjeta por modelo resume la lectura con colores:

| Elemento | Colores y criterio |
|---|---|
| Decisión | rojo = FALSO (p ≥ 0.5) · verde = REAL; confianza *alta* si p < 0.1 o p > 0.9, *poco fiable* si 0.3 ≤ p ≤ 0.7 |
| Masa por región | barra naranja (interior) · morado (borde) · azul (fuera) |
| Patrón del mapa | *interior concentrado* si masa interior ≥ 0.60 · *disperso fuera* si masa fuera ≥ 0.30 · *mixto* el resto |
| Coherencia | ✓ si la decisión coincide con lo medido en el test (falso ↔ interior, real ↔ fuera); ⚠ si no |
| Acuerdo | aviso si BCE y Focal dan decisiones distintas |

Los umbrales del patrón salen de los rangos del notebook 06 (respuesta «falso»: interior 0.67–0.95; respuesta «real»: fuera 0.31–0.62). La sección «Cómo interpretar los resultados» explica cada elemento de las tarjetas. Aparte, «Qué encontró el proyecto y límites de la demo» resume en dos tarjetas la hipótesis frente al resultado (interior, no borde de la mezcla; FaceSwap se escapa; Focal mejora solo NeuralTextures) y los límites (fotos fuera de la distribución de FF++ c23; un caso aislado no prueba nada).

## Requisitos

- Pesos `bce_s42.pt` y `focal_g2_s42.pt` (~80 MB cada uno), del output del notebook 04 v3 en Kaggle (carpeta `checkpoints/`). Se copian a `results/checkpoints/`, que Git ignora. **No se suben al repositorio:** son un derivado de FF++.
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

1. **Foto real de un integrante** (con su consentimiento). Lo esperado: «real» y masa sobre todo fuera de la cara. Probarla antes: una foto de cámara está fuera de la distribución de FF++ c23 y el modelo puede equivocarse; si pasa, sirve para hablar del cambio de distribución.
2. **Deepfake detectado:** masa concentrada en el interior de la cara.
3. **FaceSwap no detectado:** el modelo responde «real»; no aparece la evidencia del interior que aprendió de Deepfakes.
4. **Cierre:** Focal cambia poco el mapa, igual que en los números (mejora solo NeuralTextures, +0.04).

## Licencia y privacidad

- La app no guarda imágenes ni resultados. Gradio guarda temporalmente lo que se sube en su caché; la app la borra cada 10 minutos y al cerrarse.
- Las caras de FF++ solo se muestran en vivo, desde la máquina del grupo: no se graban, no se capturan y no se publican.
- No se agregan imágenes de ejemplo al repositorio.
