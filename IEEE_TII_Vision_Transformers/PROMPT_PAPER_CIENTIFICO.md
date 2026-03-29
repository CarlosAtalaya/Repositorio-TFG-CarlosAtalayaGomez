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

### 1.2 Curación del dataset (contribución metodológica; lectura obligatoria para Methodology)

La comparativa arquitectural del paper **reposa** en un único conjunto **unificado, trazable y evaluable con protocolo COCO**. No es un detalle accesorio: sin taxonomía común, balance controlado y particiones validadas, los mAP **no serían comparables** entre modelos.

**Fuente normativa en la memoria:** `TFG_Carlos_Atalaya_25-26/tex/desarrollo.tex`, sección **Preparación y curación del conjunto de datos** (etiqueta LaTeX `sec:preparacion_dataset`). El borrador `paper_vit_defect_detection.tex` ya incluye una subsección **Dataset Curation**; debe mantenerse **alineada** con esa sección de la memoria (cifras, etapas, taxonomía).

**Fuentes originales**

| Fuente | Rol en el trabajo |
|--------|-------------------|
| **VISION-Datasets** | Imágenes industriales con anotaciones tipo **COCO** (cajas); alta granularidad de etiquetas por componente. |
| **MVTec AD** | Referencia en anomalías industriales; anotación original orientada a **máscaras** y protocolo distinto; requiere conversión y mapeo al esquema unificado. |

**Pipeline en cinco etapas (Etapas 1–5)** — resumen para redacción del paper:

| Etapa | Naturaleza | Idea clave |
|-------|------------|------------|
| **1** | Análisis | Exploración de estructuras, formatos y distribución de defectos; **sin** reducir aún el volumen de forma definitiva. |
| **2** | Transformación | Curación inicial: filtrado, **PNG**, primera unificación **COCO** (orden **~1.907** imágenes, **15** etiquetas; desbalance fuerte **~24,5:1** máx/mín). |
| **3** | Análisis | Calidad: duplicados (orden **49** pares), integridad, resolución (**262×192** a **3840×3620** px), inconsistencias menores. |
| **4** | Transformación | Subetapas: eliminación de componentes fuera de alcance y duplicados (**~1.393** imágenes); **mapeo 15 → 6** categorías; **balanceo** híbrido + aumento **conservador** (**368** imágenes aumentadas); splits **70 / 10 / 20** con estratificación **dual** (clase y **origen** MVTec vs VISION). Resultado: **1.022** imágenes, **1.354** anotaciones. |
| **5** | Validación | Estratificación comprobada (p. ej. **χ²** con **p = 0,999** en la memoria), **0 fugas** entre train/val/test, integridad de ficheros. |

**Taxonomía unificada (6 categorías):** incluye **NORMAL** como clase explícita del detector (no solo fondo). Entre las fuentes existían del orden de **93** etiquetas distintas antes del mapeo; criterios: equivalencia semántica, coherencia visual y decisión metodológica documentada en la memoria.

**Implicaciones para el artículo IEEE**

- En **Introduction / Contributions**, puede explicitarse una línea del tipo: *curated multi-source industrial dataset with unified taxonomy and stratified splits* (sin sobrepasar el límite de palabras del abstract).
- En **Methodology**, el párrafo sobre dataset debe permitir a un revisor juzgar **reproducibilidad** (fuentes públicas, número final de imágenes/anotaciones, número de clases, protocolo COCO, test set fijo).
- En **Discussion / Limitations**, el tamaño (**~1k** imágenes) y el dominio acotado son límites honestos ya alineados con `paper_vit_defect_detection.tex`.
- **Figuras opcionales** para material suplementario: distribución por clase, por fuente, o evolución del proceso (si la revista lo admite).

**Código y scripts (fuera de este repositorio):** si se menciona reproducibilidad ampliada, el pipeline por etapas puede referenciarse como documentación/scripting en el proyecto de experimentación (p. ej. carpeta `dataset_preparation` del código del TFG), sin listar rutas locales en el PDF final salvo que se publique un enlace estable.

### 1.3 Capítulos y función narrativa

| Capítulo / archivo | Contenido esencial |
|--------------------|---------------------|
| `tex/resumen.tex` | Resumen y abstract bilingües; cifra clave mAP **0,785**; mensaje de **superioridad arquitectónica** (no solo resolución). |
| `tex/introduccion.tex` | Contexto industrial, CNN vs ViT, SSL, motivación, alcance (GPU RTX 4070), organización del documento y mención del **repositorio** y la **herramienta Streamlit**. |
| `tex/objetivos.tex` | Objetivo general, específicos (implementación ViT, comparativa CNN, dataset, herramienta, validación de robustez), hipótesis y contribución esperada. |
| `tex/antecedentes.tex` | Estado del arte (ViT, SSL, detección industrial, transfer learning); base para **Related Work** del paper. |
| `tex/desarrollo.tex` | **Sección larga de curación (Etapas 1–5)** + metodología por **fases experimentales** (1: CNN; 2: DEIMv2 iteraciones; 3: CNN a 1024×1024; 4: umbrales; trabajo adicional multimodal en memoria). |
| `tex/resultados.tex` | Síntesis cuantitativa, descripción **detallada del panel Streamlit** (seis vistas), figuras tipo dashboard exportadas desde la herramienta. |
| `tex/conclusiones.tex` | Números finales (ResNet **0,077–0,080**, EfficientNet **0,162–0,122**, DEIMv2 **0,785**), interpretación (atención global, DINOv3), Fase 4, líneas futuras (dataset, VLMs, etc.). |

### 1.4 Fases experimentales (alineación memoria ↔ paper)

- **Preparación del conjunto de datos (Etapas 1–5 en la memoria; “Fase 0” en sentido informal):** curación completa descrita en **§1.2**; taxonomía 6 clases; estadísticas finales (imágenes, anotaciones, splits). **Toda** la experimentación posterior usa **exclusivamente** este conjunto cerrado.
- **Fase 1 — Baseline CNN:** ResNet-18 y EfficientNet-B0 + Faster R-CNN; mAP bajo frente al detector ViT.
- **Fase 2 — DEIMv2:** iteraciones 640×640 → 1024×1024, más épocas hasta configuración óptima (**300 épocas**, mejor *checkpoint* intermedio).
- **Fase 3 — Validación:** mismas condiciones de resolución para CNN; mejora despreciable o **degradación** → la ventaja del ViT se atribuye a la **arquitectura**, no a “más píxeles” solos.
- **Fase 4 — Robustez (memoria):** evaluación bajo **umbrales** de confianza más estrictos; mAP sigue alto (p. ej. **0,785 → 0,770 → 0,705** según memoria/resultados). Útil como **extensión** del Discussion del paper si cabe en la plantilla.
- **Exploración adicional en memoria (no resumida igual en el paper):** intento **multimodal** (fusión con señal tipo CLIP) con **resultado negativo** respecto al modelo unimodal; valor como **hallazgo negativo** o anexo, no como contribución principal.

### 1.5 Cómo expresar la mejora (evitar ambigüedad)

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

- **Dataset:** contrastar el texto de **Dataset Curation** en `paper_vit_defect_detection.tex` con `TFG_Carlos_Atalaya_25-26/tex/desarrollo.tex` (sección **Preparación y curación del conjunto de datos**). Comprobar números: imágenes finales (**1.022**), anotaciones (**1.354**), clases (**6**), splits (**70/10/20**), ratio de balance (**~2,08:1**), evolución por etapas si se incluye tabla resumida.
- El **abstract del paper** menciona **DINOv2** en un párrafo sobre SSL; la memoria y el desarrollo del detector insisten en **DINOv3**. Unificar criterio y citas con el `.bib` del TFG (`TFG_Carlos_Atalaya_25-26/bib/main.bib`) al pasar a `biblatex`/BibTeX o al completar `\bibitem`.
- La taxonomía en tabla del paper usa nombres en inglés (**FRACTURE**, **SCRATCHES**); la memoria usa etiquetas tipo **ROTURA_FRACTURA**, **RAYONES_ARAÑAZOS**. Mantener **correspondencia explícita** una sola vez en Methodology.
- **Referencias:** sustituir placeholders por bibliografía completa al estilo IEEE; preferir **una sola fuente de verdad** (exportar desde `main.bib` o usar el estilo que exija TII).
- **Fase 4 y multimodal:** no están en el `.tex` del paper actual; decidir si se añaden como párrafo corto de robustez / limitación o se dejan solo en TFG.

---

## 3. Instrucciones para un agente de IA que mejore el paper

1. Leer **`paper_vit_defect_detection.tex`** y localizar huecos (placeholders, inconsistencias DINOv2/v3).
2. Leer con prioridad la sección de **curación del dataset** en **`TFG_Carlos_Atalaya_25-26/tex/desarrollo.tex`** (desde *Preparación y curación del conjunto de datos* hasta el final de ese bloque, antes de *Metodología de trabajo*). Cualquier párrafo del paper sobre el dataset debe ser coherente con esas tablas y cifras.
3. Contrastar el resto de cifras y afirmaciones con **`resultados.tex`** y **`conclusiones.tex`**.
4. No inventar métricas; si falta un dato, pedirlo o citar la sección de la memoria donde aparece.
5. Mantener **inglés académico** IEEE (voz pasiva aceptable, frases directas, sin adorno).
6. Priorizar claridad: **problema → datos y protocolo → método de entrenamiento → resultados principales → limitaciones**.
7. Al ampliar Discussion, considerar aportar **coste computacional**, **tiempo de inferencia** y **generalización** solo si hay números en la memoria o en anexos.

---

## 4. Redacción al estilo IEEE (recomendaciones prácticas)

Basado en el [IEEE Author Center](https://journals.ieeeauthorcenter.ieee.org/) y práctica habitual en revistas tipo **IEEE Transactions**:

### 4.1 Estructura típica (journals)

- **Abstract:** un solo párrafo (suele haber límite de palabras, a menudo **150–250**). **Autocontenido:** sin ecuaciones, sin citas, sin referencias a “Fig. 1”. Definir siglas la primera vez si son inevitables.
- **Introduction:** contexto amplio → problema técnico → breve panorama de enfoques → **hueco** (*gap*) → **contribuciones** (lista numerada o con viñetas al final de la sección).
- **Related Work:** agrupado por **temas** (p. ej. ViT en inspección, SSL, detectores tipo DETR), no solo lista de papers; al final, una frase que diga qué **no** cubre la literatura y qué hace este trabajo.
- **Methodology:** dataset (**origen de las fuentes**, **pasos de curación** suficientes para reproducibilidad, taxonomía, **balanceo**, **splits** y métricas de evaluación), arquitecturas, entrenamiento, **protocolo reproducible** (hardware opcional pero útil).
- **Experiments / Results:** tablas y figuras con **subtítulos claros**; comparaciones **justas** (mismos datos, mismas métricas).
- **Discussion:** interpretación, no repetir números sin sentido; limitaciones honestas.
- **Conclusion:** breve; a veces se fusiona con Discussion según plantilla.

### 4.2 Estilo y forma

- Oraciones **cortas y precisas**; un idea principal por frase en abstract e introduction.
- **Primera persona plural** (“we”) o **voz pasiva** son aceptables; evitar mezclar sin criterio.
- **Siglas:** definir en abstract (si la revista lo permite) o al primer uso en el cuerpo.
- **Figuras y tablas:** citarlas en el texto en orden; leyendas **autocontenidas** (el lector debe entender la figura sin leer todo el párrafo).
- **Novedad:** el revisor busca **qué cambia** respecto a “aplicar un ViT genérico”; dejar explícitas las contribuciones (comparación sistemática bajo **un mismo dataset curado**, control de resolución en Fase 3, protocolo COCO, etc.).

### 4.3 Detalles que suelen pedir las revistas IEEE

- Plantilla oficial LaTeX/Word de la revista concreta (**TII** tiene guía propia junto a la plantilla `TII-Articles-LaTeX-template`).
- **Keywords** alineadas con IEEE Xplore / tesauro del dominio.
- Figuras en **vectorial** cuando sea posible (PDF/EPS); resolución suficiente en mapas de bits.
- **Datos y código:** según política de la revista (repositorio suplementario, material multimedia).

### 4.4 Transparencia (2024 en adelante)

- Algunas revistas IEEE piden declaración sobre **herramientas de IA** en redacción o figuras; revisar la guía actual de **IEEE TII** al enviar.

---

## 5. Tablas resumen (comprobación rápida)

### 5.1 Dataset curado (tras Etapas 1–5)

| Magnitud | Valor orientativo (memoria) |
|----------|-----------------------------|
| Imágenes finales | **1.022** |
| Anotaciones | **1.354** |
| Categorías | **6** |
| Particiones | **70 % / 10 % / 20 %** (train / val / test) |
| Ratio máx/mín post-balanceo | **~2,08 : 1** |
| Aumento sintético (imágenes) | **368** (política conservadora) |
| Test fijo para todos los modelos | **205** imágenes (misma partición) |

*Confirmar en `TFG_Carlos_Atalaya_25-26/tex/desarrollo.tex` si alguna cifra se revisó en correcciones finales.*

### 5.2 Resultados comparativos (mAP@0.5)

| Arquitectura (mejor config. citada) | mAP@0.5 (orden de magnitud) |
|-----------------------------------|----------------------------|
| ResNet-18 + Faster R-CNN | ~0,077–0,080 |
| EfficientNet-B0 + Faster R-CNN | ~0,122–0,162 (nativa vs 1024) |
| DEIMv2 (DINOv3), 1024×1024, 300 ép. | **0,785** |

*Confirmar valores finales en `tex/desarrollo.tex` / tablas del paper antes de envío.*

---

## 6. Cómo empezar una sesión de edición

1. Abrir `paper_vit_defect_detection.tex` y la plantilla `TII-Articles-LaTeX-template/tii-articles-template.tex` por si hay requisitos adicionales de secciones.
2. Revisar primero la subsección **Dataset Curation / Methodology** del paper frente a **`TFG_Carlos_Atalaya_25-26/tex/desarrollo.tex`** (bloque *Preparación y curación del conjunto de datos*).
3. Abrir en paralelo `introduccion.tex`, `antecedentes.tex`, `resultados.tex`, `conclusiones.tex` según la sección a editar.
4. Sustituir **placeholders** de bibliografía y unificar **DINOv2/v3** según el trabajo real.
5. Opcional: un párrafo o nota sobre **herramienta de análisis** (Streamlit) o **datos suplementarios** (distribución por clase/origen), sin inflar el artículo principal.

---

*Última actualización orientativa: marzo 2026. Ajustar fechas y políticas de la revista al momento del envío.*
