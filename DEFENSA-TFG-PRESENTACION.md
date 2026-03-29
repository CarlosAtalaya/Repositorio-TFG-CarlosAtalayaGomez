# Propuesta de estructura para la defensa oral del TFG (presentación + demo interactiva)

**Audiencia objetivo:** tribunal de la Escuela de Ingeniería con formación técnica general, **sin necesidad de ser experto en visión por computador o aprendizaje profundo**. El mensaje debe ser: **qué problema industrial hay**, **sobre qué datos fiables se apoya el estudio**, **qué se ha hecho**, **por qué importa**, **qué resultado objetivo se obtiene**, y **cómo se comprueba**.

**Recursos disponibles:**

- Memoria cerrada en `TFG_Carlos_Atalaya_25-26/` (Capítulo de desarrollo: **Preparación y curación del conjunto de datos** y fases experimentales).
- **Herramienta Streamlit** (`Documentacion-externa/FASE-1-analisis/herramienta_comparativa/`, `dashboard.py`): panel con vistas **Inicio**, **Línea temporal**, **Explorador**, **Comparativa**, **Visualizaciones**, **Conclusiones**.
- **Pipeline y documentación de curación** (código y guías fuera de este repo, si el tribunal pregunta por reproducibilidad):  
  `.../ViT/ViT-Industrial-Defects/dataset_preparation/` (`DOCUMENTACION_CURACION_DATASET.md`, `flujo_curacion_dataset/`, scripts por etapa).

**Principio de diseño:** la presentación lleva la **historia**; la curación del dataset se presenta como **condición necesaria** para que la comparativa CNN vs ViT sea **justa y defendible** (no como un segundo TFG dentro del TFG). Streamlit lleva la **evidencia interactiva** de **experimentos**; las **figuras de la memoria** (distribución por clase, por fuente, tamaños de caja) son el mejor apoyo visual para el bloque de datos.

**Tope de tiempo:** **15–17 minutos en total** (todo lo que hablas y enseñas en pantalla **antes** de la ronda de preguntas). No hay margen para una demo larga: la demo debe ser **breve y ensayada** o sustituirse por **capturas fijas** en el propio PPT.

**Reparto orientativo (encaja en 15 o 17 min):**

| Tramo | 15 min (reparto ajustado) | 17 min (un poco más holgado) |
|-------|---------------------------|-------------------------------|
| Apertura (título + problema + pregunta) | ~1,5 min | ~2 min |
| Dataset curado (mensaje + cifras + 1 figura) | ~3 min | ~3 min |
| Enfoque y experimentos (CNN vs ViT, DEIMv2, fases) | ~3 min | ~3,5 min |
| Resultados (mAP + Fase 3) | ~2 min | ~2,5 min |
| Demo Streamlit **o** lámina con capturas | **2 min** | **3 min** |
| Conclusiones y cierre | ~1,5 min | ~1,5 min |
| **Colchón** (respirar, transiciones) | ~2 min | ~1,5 min |

*(Suma orientativa: ~15 min y ~17 min respectivamente.)*

Si vas justo de tiempo: **elimina la demo en vivo** y usa **una sola lámina** con captura de **Comparativa** + **Visualizaciones** (60–90 s de comentario).

**Prioridad si hay que recortar (en este orden):** (1) robustez por umbrales, (2) demo en vivo, (3) segunda figura del dataset, (4) detalle de fases 1–4 en láminas (sustituir por una frase y pasar rápido a resultados). **No recortes** el mensaje de “datos comparables → comparativa válida” (mínimo **~2,5–3 min** entre datos + figura).

---

## 1. Mensaje central sobre el dataset (para no pasarse y no quedarse corto)

**Idea única que debe quedar:** *Los resultados comparativos solo son creíbles si todos los modelos entrenan y evalúan sobre el **mismo** conjunto bien definido: fuentes unificadas, **taxonomía coherente**, **balance controlado**, **particiones sin fugas** y **métricas COCO**. Esa preparación fue un trabajo experimental explícito (Etapas 1–5), no un simple “descarga y entrena”.*

**Evitar:** entrar en nombres de scripts, rutas de carpetas o hiperparámetros de aumento en voz alta.

**Sí incluir:** una **cifra memorable** (p. ej. de **93 etiquetas originales** a **6 categorías**; o ratio de desbalance **~24,5:1 → 2,08:1**) y **una** figura de la memoria (barras por clase o por fuente MVTec vs VISION).

---

## 2. Síntesis técnica (referencia interna para ti; no leer al tribunal)

Contenido alineado con `tex/desarrollo.tex` (sección **Preparación y curación del conjunto de datos**) y con la documentación del pipeline en `dataset_preparation/`.

| Idea | Dato orientativo (memoria) |
|------|----------------------------|
| Fuentes | **VISION-Datasets** (COCO, muchas etiquetas finas) + **MVTec AD** (referencia en anomalías; máscaras / filosofía distinta). |
| Problema inicial | Formatos y granularidad distintos; **decenas de etiquetas** en las fuentes combinadas. |
| Pipeline | **5 etapas**: exploración → curación inicial COCO → análisis de calidad → recuración (limpieza, taxonomía, balanceo, splits) → validación estadística. |
| Evolución de volumen | Del orden de **~9,1k** imágenes combinadas en bruto a **1.022** imágenes finales; de **64** a **6** categorías en la tabla de evolución. |
| Taxonomía | **6 clases** (incluye **NORMAL** como clase explícita del detector, no solo “fondo”). Reducción desde **93** etiquetas distintas entre fuentes, con criterios de equivalencia semántica y coherencia visual. |
| Calidad / calidad de anotaciones | Detección de **duplicados**, control de integridad, caracterización de **resolución** (muy variable) y de **cajas pequeñas** (relevante para sensibilidad del detector y para justificar experimentos a distinta resolución). |
| Balance | Desbalance fuerte tras curación inicial (orden **24,5:1**); tras balanceo híbrido + aumento **conservador**, ratio **~2,08:1**. |
| Particiones | **70 / 10 / 20** (entrenamiento / validación / prueba), estratificación **dual** (categoría y origen de fuente). |
| Validación | **Chi-cuadrado** sobre estratificación (**p = 0,999** en memoria), **0 fugas** entre splits, integridad de ficheros. |
| Aumento | **368** imágenes aumentadas; transformaciones suaves (volteo, rotación pequeña, brillo/contraste leve, ruido), sin deformaciones agresivas que rompan el defecto. |

**Conexión con resultados:** el mismo test (**205 imágenes**) y el mismo protocolo COCO para **todas** las arquitecturas; por eso la ventaja del ViT se puede atribuir al modelo cuando además la **Fase 3** controla la resolución.

---

## 3. Guion general (storyline) — ajustado a 15–17 min

Tiempos son **orientativos**; ensaya con cronómetro.

1. **Gancho (~30 s):** inspección = calidad y coste; pocos defectos etiquetados; sin criterio de datos unificado, **las comparativas entre modelos no valen**.
2. **Problema y pregunta (~45–60 s):** detección con cajas y clases; **¿ViT puede superar a CNN con datos limitados bajo el mismo protocolo?**
3. **Bloque datos (~3 min):** VISION + MVTec → **COCO** y **6 categorías** (de muchas etiquetas a taxonomía única); **1.022** imágenes, splits, balance **~2:1**, sin fugas; **una figura** (barras por clase o por fuente).
4. **Modelo y experimentos (~3 min):** CNN vs ViT en una frase; **DEIMv2 + DINOv3**; en **una lámina** o frase oral: Fase 1 CNN → Fase 2 ViT → Fase 3 misma resolución para CNN → (opcional) Fase 4 umbrales **solo si sobran 30 s**.
5. **Resultados (~2 min):** mAP (~0,785 vs CNN muy por debajo); **Fase 3:** la ventaja **no** es solo “más píxeles”.
6. **Demo (~2–3 min)** o **capturas en PPT (~90 s):** Comparativa + **una** imagen en Visualizaciones; no navegar más vistas.
7. **Cierre (~1,5 min):** tres aportaciones (datos, resultado, herramienta); una limitación; gracias.

---

## 4. Mapa diapositiva ↔ demo Streamlit / figuras memoria

| Bloque PPT | Contenido mínimo en lámina | Apoyo visual recomendado |
|------------|---------------------------|---------------------------|
| Título | Título del TFG, nombre, tutor/es, escuela | — |
| Motivación | Inspección, coste del error, datos escasos | Icono / foto industrial |
| **Por qué el dataset importa** | Una frase: “sin criterio común de etiquetas y particiones, la comparativa no vale” | Opcional: icono “datos → modelo → métrica” |
| **Fuentes y reto** | VISION + MVTec; formatos distintos; muchas etiquetas en bruto | Logo o nombres de datasets (sin saturar) |
| **Pipeline en 5 pasos** | Esquema horizontal: explorar → curar → analizar → recurar (taxonomía + balance + splits) → validar | **Figura del memoria** `Diagrama-inicial-section-desarrollo.png` o esquema propio equivalente |
| **Qué se obtuvo** | **1.022** imágenes, **1.354** anotaciones, **6** clases; splits **70/10/20**; balance **~2:1**; **p** estratificación | **Figura** `fase0_category_distribution.png` o `fase0_source_dataset_distribution.png` |
| **Detalle que engancha con experimentos** *(opcional si vas muy justo: omitir)* | Variación de resolución y **cajas pequeñas** → motiva probar **640 vs 1024** | Solo si cabe en 15 min: miniatura o una frase en la lámina de resultados |
| Contexto ML | Imagen → detector → cajas; CNN vs ViT (sin ecuaciones) | — |
| Enfoque | DEIMv2 + DINOv3 (SSL) | Diagrama arquitectura del TFG si lo tienes |
| Diseño experimental | Fases 1–4 (entrenamiento); recordar que **Fase 0** = datos (ya explicada) | **Línea temporal** en Streamlit |
| Resultados | mAP: CNN vs ViT | **Comparativa** Streamlit o figura barras memoria |
| Arquitectura vs resolución | Fase 3 | **Explorador** |
| Robustez *(solo 17 min o si preguntan)* | Umbrales | **Explorador** o una frase oral |
| Demo cualitativa | “Qué ve cada modelo” | **Visualizaciones** |
| Herramienta | Panel de análisis | Captura o **Inicio** |
| Conclusiones + preguntas | 3 viñetas + agradecimiento | — |

---

## 5. Estructura detallada para 15–17 minutos (láminas + tiempos)

**Objetivo:** **10–12 diapositivas** de contenido (más título y gracias). Menos láminas = más tiempo por mensaje.

### Apertura — ~1,5–2 min — **1–2 láminas**

1. **Título** (nombre, tutor, título del TFG).
2. **Problema + pregunta en la misma lámina** (recomendado para ahorrar tiempo): inspección industrial, datos escasos, y *¿ViT supera a CNN bajo el mismo dataset y métricas?*

### Bloque “Dataset curado” — ~3–3,5 min — **2 láminas** (no cuatro)

3. **Datos: fuentes + proceso (combinada):** VISION + MVTec; reto de unificar; **pipeline en 5 etapas** en viñetas muy cortas (una línea cada una) **o** una sola figura tipo `Diagrama-inicial-section-desarrollo.png`.
4. **Datos: resultado + prueba:** **1.022** imágenes, **6** clases, **70/10/20**, balance **~2:1**, sin fugas, χ²; **una figura** (`fase0_category_distribution` o `fase0_source_dataset_distribution`).

### Núcleo modelo + experimentos — ~3–3,5 min — **2–3 láminas**

5. **Tarea + intuición CNN vs ViT** (puede ser una sola lámina: esquema imagen → cajas).
6. **DEIMv2 + DINOv3** (1 lámina breve).
7. **Fases experimentales** (1 lámina): línea temporal **o** lista compacta F1→F2→F3 (F4 umbrales **opcional** y solo si usas 17 min).

### Resultados — ~2–2,5 min — **2 láminas**

8. **mAP:** mejor CNN vs mejor DEIMv2 (gráfico de barras de la memoria o captura de **Comparativa**).
9. **Fase 3** en media lámina o lámina corta: misma resolución 1024 para CNN → la brecha **no** se explica solo por resolución.

### Demo — **2 min (15 min total)** o **3 min (17 min total)**

10. **Una lámina** “Evidencia” con título fijo.
    - **En vivo:** abrir ya la vista **Comparativa** antes de hablar; mostrar **10–15 s** el gráfico global; cambiar a **Visualizaciones** con **una imagen ensayada**; **no** pasar por Inicio, Línea temporal ni Conclusiones.
    - **Sin tiempo:** misma lámina con **capturas estáticas** de esas dos vistas y comentario de 60–90 s.

### Cierre — ~1,5–2 min — **1 lámina**

11. **Conclusiones + limitaciones + gracias / preguntas** (todo en una lámina con 3 viñetas + una línea de limitación).

**Ensayo:** cronómetro en móvil; si pasas de **16 min** en el ensayo general, recorta Fase 4, robustez y la segunda figura del dataset.

---

## 6. Consejos para el tribunal (no experto en visión)

- **“Curación”** en una frase: *unificar criterios de etiqueta, limpiar duplicados, equilibrar clases y separar bien entrenamiento y prueba para que el número final del modelo sea interpretable*.
- **mAP** en una frase: *puntuación global de acierto del detector en todas las clases* (detalle IoU solo si preguntan).
- Conectar **cajas pequeñas** del dataset con la **motivación** de probar **diferentes resoluciones** de entrada (sin tecnicismo de *anchors* salvo pregunta).
- Si preguntan **reproducibilidad**: citar que el flujo está documentado y scriptado (`dataset_preparation`, etapas numeradas), sin abrir el portátil a esa ruta en la sala salvo que quieras.

---

## 7. Checklist previa al día de la defensa (15–17 min)

- [ ] Ensayo **completo con cronómetro** al menos **2 veces**; objetivo: **≤ 16 min** dejando 1 min de margen.
- [ ] Bloque dataset **≤ 3,5 min** en voz alta (si pasas de 4 min, fusiona láminas 3 y 4).
- [ ] **Plan A:** Streamlit con **Comparativa** precargada y **Visualizaciones** en la imagen acordada (ventana lista **antes** de tu turno si la sala lo permite).
- [ ] **Plan B:** misma información en **2 capturas** en el PPT (sin demo en vivo).
- [ ] **1 figura** de datos en el PPT (no tres); el resto solo si sobra tiempo en el ensayo.

---

## 8. Ideas visuales (PPT “clara y seria”)

- Misma **plantilla gráfica** que la memoria o el Streamlit (coherencia).
- **Un diagrama de flujo** horizontal: Etapa 1 → … → Etapa 5 (reutilizable en papel y en defensa).
- Icono fijo para “dato curado” (p. ej. checklist) vs “modelo” (p. ej. chip) para que el tribunal siga el hilo.

---

*Documento de apoyo para preparar la defensa con **tope 15–17 minutos** totales. El bloque de dataset debe sonar a **contribución metodológica** en **~3 minutos**, no a un segundo proyecto.*

