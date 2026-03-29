# Contexto y guía para el artículo IEEE (TII) — Vision Transformers en detección industrial

Este documento sirve para **contextualizar** el trabajo publicable frente a la memoria del TFG, indicar **qué está ya redactado** en el repositorio, **qué revisar antes de enviar**, y **cómo redactar** al estilo IEEE. Está pensado para humanos y para agentes de IA que ayuden a mejorar el manuscrito.

---

## 1. Sinopsis de la memoria técnica (TFG)

**Ubicación de la fuente:** `TFG_Carlos_Atalaya_25-26/` (documento principal `TFG.tex`).

### 1.1 Tema y narrativa

- **Problema:** inspección visual industrial con **pocos defectos etiquetados** (*small data*) y limitaciones de las **CNN** para captar **contexto global** y defectos sutiles o milimétricos.
- **Propuesta:** detector **DEIMv2** con **backbone ViT** preentrenado con **aprendizaje auto-supervisado (DINOv3)** frente a líneas base **ResNet-18** y **EfficientNet-B0** con **Faster R-CNN**.
- **Datos:** conjunto **curado y unificado** a partir de **VISION-Datasets** y **MVTec AD**, formato **COCO**, **6 categorías** semánticas, **1.022 imágenes**, particiones sin fugas, balance comprobado (p. ej. contraste χ²).
- **Métricas:** protocolo **COCO** (principalmente **mAP@0.5**, AP por clase, precisión, exhaustividad), análisis de **umbrales de confianza** para robustez.
- **Herramienta:** **panel Streamlit** (Explorador, Comparativa, Visualizaciones, etc.) para auditoría e interpretabilidad de resultados.

### 1.2 Capítulos y función narrativa

| Capítulo / archivo | Contenido esencial |
|--------------------|---------------------|
| `tex/resumen.tex` | Resumen y abstract bilingües; cifra clave mAP **0,785**; mensaje de **superioridad arquitectónica** (no solo resolución). |
| `tex/introduccion.tex` | Contexto industrial, CNN vs ViT, SSL, motivación, alcance (GPU RTX 4070), organización del documento y mención del **repositorio** y la **herramienta Streamlit**. |
| `tex/objetivos.tex` | Objetivo general, específicos (implementación ViT, comparativa CNN, dataset, herramienta, validación de robustez), hipótesis y contribución esperada. |
| `tex/antecedentes.tex` | Estado del arte (ViT, SSL, detección industrial, transfer learning); base para **Related Work** del paper. |
| `tex/desarrollo.tex` | Metodología por **fases** (0: dataset; 1: CNN; 2: DEIMv2 iteraciones; 3: CNN a 1024×1024; 4: umbrales; trabajo adicional multimodal en memoria). |
| `tex/resultados.tex` | Síntesis cuantitativa, descripción **detallada del panel Streamlit** (seis vistas), figuras tipo dashboard exportadas desde la herramienta. |
| `tex/conclusiones.tex` | Números finales (ResNet **0,077–0,080**, EfficientNet **0,162–0,122**, DEIMv2 **0,785**), interpretación (atención global, DINOv3), Fase 4, líneas futuras (dataset, VLMs, etc.). |

### 1.3 Fases experimentales (alineación memoria ↔ paper)

- **Fase 0 — Dataset:** curación, taxonomía 6 clases, estadísticas finales (imágenes, anotaciones, splits).
- **Fase 1 — Baseline CNN:** ResNet-18 y EfficientNet-B0 + Faster R-CNN; mAP bajo frente al detector ViT.
- **Fase 2 — DEIMv2:** iteraciones 640×640 → 1024×1024, más épocas hasta configuración óptima (**300 épocas**, mejor *checkpoint* intermedio).
- **Fase 3 — Validación:** mismas condiciones de resolución para CNN; mejora despreciable o **degradación** → la ventaja del ViT se atribuye a la **arquitectura**, no a “más píxeles” solos.
- **Fase 4 — Robustez (memoria):** evaluación bajo **umbrales** de confianza más estrictos; mAP sigue alto (p. ej. **0,785 → 0,770 → 0,705** según memoria/resultados). Útil como **extensión** del Discussion del paper si cabe en la plantilla.
- **Exploración adicional en memoria (no resumida igual en el paper):** intento **multimodal** (fusión con señal tipo CLIP) con **resultado negativo** respecto al modelo unimodal; valor como **hallazgo negativo** o anexo, no como contribución principal.

### 1.4 Cómo expresar la mejora (evitar ambigüedad)

- En la **memoria** se usa a veces **+62,3 puntos porcentuales** de mAP frente a la mejor CNN en configuración comparable (diferencia absoluta entre tasas 0–1).
- En el **borrador del paper** aparecen **porcentajes relativos** respecto a baselines muy bajos (**+881 %** vs ResNet-18, **+384 %** vs EfficientNet-B0). Son matemáticamente correctos pero **muy sensibles** a la línea base cercana a cero.
- **Recomendación para revisión/presentación:** reportar siempre **mAP absoluto**, **diferencia absoluta** y, si se usan relativos, **justificar** que la baseline es baja (o dar ambos en tabla).

---

## 2. Qué hay ya integrado en `IEEE_TII_Vision_Transformers/`

### 2.1 Archivos relevantes

| Recurso | Descripción |
|---------|-------------|
| `paper_vit_defect_detection.tex` | Borrador **completo en inglés**: título, abstract, palabras clave, Introduction (con contribuciones enumeradas), Related Work, Methodology (dataset, arquitecturas, protocolo, fases 1–3), Experiments and Results (tablas), Discussion, Conclusion, **referencias manuales** (`thebibliography`) con entradas **marcadas como PLACEHOLDER** (“To be completed”). |
| `TII-Articles-LaTeX-template/` | Plantilla oficial: `ieeecolor.cls`, `generic.sty`, `tii-articles-template.tex` de referencia. |
| `PROMPT_PAPER_CIENTIFICO.md` | Este documento (guía de contexto). |

### 2.2 Coherencia con la memoria a revisar por un agente

- El **abstract del paper** menciona **DINOv2** en un párrafo sobre SSL; la memoria y el desarrollo del detector insisten en **DINOv3**. Unificar criterio y citas con el `.bib` del TFG (`TFG_Carlos_Atalaya_25-26/bib/main.bib`) al pasar a `biblatex`/BibTeX o al completar `\bibitem`.
- La taxonomía en tabla del paper usa nombres en inglés (**FRACTURE**, **SCRATCHES**); la memoria usa etiquetas tipo **ROTURA_FRACTURA**, **RAYONES_ARAÑAZOS**. Mantener **correspondencia explícita** una sola vez en Methodology.
- **Referencias:** sustituir placeholders por bibliografía completa al estilo IEEE; preferir **una sola fuente de verdad** (exportar desde `main.bib` o usar el estilo que exija TII).
- **Fase 4 y multimodal:** no están en el `.tex` del paper actual; decidir si se añaden como párrafo corto de robustez / limitación o se dejan solo en TFG.

---

## 3. Instrucciones para un agente de IA que mejore el paper

1. Leer **`paper_vit_defect_detection.tex`** y localizar huecos (placeholders, inconsistencias DINOv2/v3).
2. Contrastar cifras y afirmaciones con **`TFG_Carlos_Atalaya_25-26/tex/desarrollo.tex`**, **`resultados.tex`** y **`conclusiones.tex`**.
3. No inventar métricas; si falta un dato, pedirlo o citar la sección de la memoria donde aparece.
4. Mantener **inglés académico** IEEE (voz pasiva aceptable, frases directas, sin adorno).
5. Priorizar claridad: **problema → método → resultados principales → limitaciones**.
6. Al ampliar Discussion, considerar aportar **coste computacional**, **tiempo de inferencia** y **generalización** solo si hay números en la memoria o en anexos.

---

## 4. Redacción al estilo IEEE (recomendaciones prácticas)

Basado en el [IEEE Author Center](https://journals.ieeeauthorcenter.ieee.org/) y práctica habitual en revistas tipo **IEEE Transactions**:

### 4.1 Estructura típica (journals)

- **Abstract:** un solo párrafo (suele haber límite de palabras, a menudo **150–250**). **Autocontenido:** sin ecuaciones, sin citas, sin referencias a “Fig. 1”. Definir siglas la primera vez si son inevitables.
- **Introduction:** contexto amplio → problema técnico → breve panorama de enfoques → **hueco** (*gap*) → **contribuciones** (lista numerada o con viñetas al final de la sección).
- **Related Work:** agrupado por **temas** (p. ej. ViT en inspección, SSL, detectores tipo DETR), no solo lista de papers; al final, una frase que diga qué **no** cubre la literatura y qué hace este trabajo.
- **Methodology:** dataset (origen, filtrado, splits), arquitecturas, entrenamiento, métricas, **protocolo reproducible** (hardware opcional pero útil).
- **Experiments / Results:** tablas y figuras con **subtítulos claros**; comparaciones **justas** (mismos datos, mismas métricas).
- **Discussion:** interpretación, no repetir números sin sentido; limitaciones honestas.
- **Conclusion:** breve; a veces se fusiona con Discussion según plantilla.

### 4.2 Estilo y forma

- Oraciones **cortas y precisas**; un idea principal por frase en abstract e introduction.
- **Primera persona plural** (“we”) o **voz pasiva** son aceptables; evitar mezclar sin criterio.
- **Siglas:** definir en abstract (si la revista lo permite) o al primer uso en el cuerpo.
- **Figuras y tablas:** citarlas en el texto en orden; leyendas **autocontenidas** (el lector debe entender la figura sin leer todo el párrafo).
- **Novedad:** el revisor busca **qué cambia** respecto a “aplicar un ViT genérico”; dejar explícitas las contribuciones (comparación sistemática, control de resolución, protocolo COCO en dataset unificado, etc.).

### 4.3 Detalles que suelen pedir las revistas IEEE

- Plantilla oficial LaTeX/Word de la revista concreta (**TII** tiene guía propia junto a la plantilla `TII-Articles-LaTeX-template`).
- **Keywords** alineadas con IEEE Xplore / tesauro del dominio.
- Figuras en **vectorial** cuando sea posible (PDF/EPS); resolución suficiente en mapas de bits.
- **Datos y código:** según política de la revista (repositorio suplementario, material multimedia).

### 4.4 Transparencia (2024 en adelante)

- Algunas revistas IEEE piden declaración sobre **herramientas de IA** en redacción o figuras; revisar la guía actual de **IEEE TII** al enviar.

---

## 5. Tabla resumen de números (comprobación rápida)

| Arquitectura (mejor config. citada) | mAP@0.5 (orden de magnitud) |
|-----------------------------------|----------------------------|
| ResNet-18 + Faster R-CNN | ~0,077–0,080 |
| EfficientNet-B0 + Faster R-CNN | ~0,122–0,162 (nativa vs 1024) |
| DEIMv2 (DINOv3), 1024×1024, 300 ép. | **0,785** |

*Confirmar valores finales en `tex/desarrollo.tex` / tablas del paper antes de envío.*

---

## 6. Cómo empezar una sesión de edición

1. Abrir `paper_vit_defect_detection.tex` y la plantilla `TII-Articles-LaTeX-template/tii-articles-template.tex` por si hay requisitos adicionales de secciones.
2. Abrir en paralelo `TFG_Carlos_Atalaya_25-26/tex/introduccion.tex`, `antecedentes.tex`, `desarrollo.tex`, `conclusiones.tex`.
3. Sustituir **placeholders** de bibliografía y unificar **DINOv2/v3** según el trabajo real.
4. Opcional: añadir un párrafo sobre **herramienta de análisis** (Streamlit) como material suplementario o reproducibilidad, sin inflar el artículo principal.

---

*Última actualización orientativa: marzo 2026. Ajustar fechas y políticas de la revista al momento del envío.*
