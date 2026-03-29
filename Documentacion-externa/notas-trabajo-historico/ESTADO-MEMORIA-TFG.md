# Estado de la Memoria TFG — Carlos Atalaya Gómez
**Última actualización:** Marzo 2026  
**Título del TFG:** Detección de Anomalías Industriales con Vision Transformers

---

## 1. Tareas completadas en las últimas sesiones de edición

### 1.1 Contenido técnico (Capítulos 3, 4 y 6)
- [x] Añadida sección 3.2 con detalle técnico de ViT: fórmulas de Attention y MHSA, comparativa arquitectural CNN vs ViT.
- [x] Añadida descripción técnica de DINOv3: objetivos de preentrenamiento DINO+iBOT+DKoleo, mecanismo Gram Anchoring con fórmula.
- [x] Añadida nueva sección 3.8: DEIMv2 con detalle arquitectónico (backbone ViT/DINOv3, STA, decoder, Dense O2O).
- [x] Expandida sección de métricas de evaluación con fórmulas de IoU, Precision, Recall, AP y mAP@0.5.
- [x] Añadida subsección 4.5.2 con justificación de hiperparámetros para DEIMv2 (LR, backbone LR, AdamW, batch size, resolución).
- [x] Añadida subsección de justificación de hiperparámetros para arquitecturas CNN (ResNet-18 + SGD, EfficientNet-B0 + AdamW).
- [x] Añadida justificación del umbral de confianza 0.15 y de la precision = 1.0 en DEIMv2.
- [x] Reescrito capítulo 6: separado en Conclusiones y Trabajos a Futuro con contenido académico completo.
- [x] Añadida entrada bibliográfica de Qwen3-VL en bib/main.bib.

### 1.2 Formato y maquetación
- [x] Cabeceras: configurado `fancyhdr` en `sty/eiiatfg.cls`; número de página siempre arriba a la derecha, nombre de capítulo en la izquierda (sin mostrar secciones).
- [x] Lista de acrónimos: añadido paso `makeglossaries` en `compile.sh`; la lista de siglas ahora se genera correctamente.
- [x] Acrónimos nuevos añadidos: AP, IoU, TP, FP, FN, DEIMv2, DINOv2, DINOv3, RPN, DETR, STA, VLM, CLIP.
- [x] Tabla 3.1 y Tabla 4.1: caption movido de encima a debajo de la tabla.
- [x] Tabla 3.1 y Tabla 4.1: convertidas a `tabularx` para ajustarse al ancho del texto sin overflow.
- [x] Tabla 4.1: columna abreviada "Categ." corregida a "Categorías".
- [x] Tabla 4.15 (resumen experimentos): eliminada columna "Observaciones".
- [x] Tabla 4.17: movida para aparecer después de la lista de implicaciones.
- [x] Ecuación MHSA (línea): dividida en dos líneas con entorno `split` para evitar overflow.
- [x] `amssymb` añadido a `sty/eiiatfg.cls` para corregir errores de `\mathbb`.
- [x] Referencia al nombre de fichero `curated_dataset_splitted_...` eliminada de sección 4.2.1.
- [x] Negritas inapropiadas en cuerpo de texto eliminadas en desarrollo.tex, resultados.tex y conclusiones.tex.
- [x] Enumeraciones (1)(2)(3)(4) convertidas a entornos `enumerate[label=(\arabic*)]` en resultados.tex.
- [x] `ROTURA\_FRACTURA` y `RAYONES\_ARAÑAZOS` protegidos con `\mbox{}` en todos los ficheros para evitar partición con guiones.
- [x] "Sección" con mayúscula corregida a minúscula en todos los contextos de referencia cruzada en mid-sentence.
- [x] Separador `---` entre Conclusiones y Trabajos a Futuro eliminado de conclusiones.tex.
- [x] Em-dashes en contextos con nombres de clase larga reemplazados por paréntesis para evitar overflow.

### 1.3 Alineación de objetivos
- [x] `tex/objetivos.tex` actualizado: F1-score y AUC-ROC eliminados del objetivo 2 y reemplazados por métricas realmente usadas (mAP, AP, Precision, Recall).
- [x] Objetivo 3 reformulado: "Validación en datasets benchmark" → "Construcción y validación de dataset industrial curado".
- [x] Objetivo 4 reformulado: "Demostración con componentes capturados" → "Desarrollo de herramienta de análisis e interpretabilidad (dashboard Streamlit)".
- [x] Objetivo 5 reformulado: "Análisis de interpretabilidad por atención" → "Validación de robustez y análisis comparativo final".
- [x] Alcance actualizado: refleja el dataset curado real (1.022 imágenes, 6 categorías).

### 1.4 Pulido de formato y estructura narrativa (marzo 2026)

Objetivo: llevar la memoria a calidad de documento científico-técnico de investigación en visión por computador, facilitando el hilo narrativo.

**Enumeraciones y listas:**
- [x] Convertidas todas las enumeraciones inline tipo "(1)...(2)...(3)" a entornos `enumerate` con ítems que empiezan en mayúscula (antecedentes.tex, desarrollo.tex, resultados.tex, conclusiones.tex).
- [x] Eliminado el formato `[label=(\arabic*)]` en resultados.tex; ahora se usa enumerate estándar (1., 2., 3.).

**Em-dashes y guiones:**
- [x] Eliminados todos los em-dashes (`—`, `---`) en texto corriente; sustituidos por paréntesis, comas o dos puntos (evita overflow de margen derecho).

**Títulos y maquetación:**
- [x] Título largo sección 3.2: `\section[Fundamentos de ViT y aprendizaje auto-supervisado]{...}` para evitar desborde de margen.
- [x] Subetapas 4.1–4.4: sustituido `---` por `.` en etiquetas de lista.

**Negritas:**
- [x] Eliminadas negritas en texto corriente (solo se usan en títulos, encabezados y etiquetas estructurales). Ejemplos corregidos: "experimento crucial", "mismas condiciones", "DEIMv2", "Etapa X", "la reducción de la pérdida no garantiza...", etc.

**Estructura de contenido:**
- [x] Introducción a ResNet-18 y EfficientNet-B0 trasladada de desarrollo.tex a antecedentes.tex (nueva subsección 3.2.2). Desarrollo.tex ahora referencia `\ref{subsec:cnn-arquitecturas-referencia}`.

**Pies de figura:**
- [x] ViT, DINOv3, DEIMv2, ResNet-18, EfficientNet-B0: captions reducidos a una línea (nombre + cita). La explicación técnica está en el cuerpo de la memoria.

**Validación técnica:**
- [x] Fórmulas matemáticas (Attention, MHSA, IoU, Precision, Recall, AP) verificadas y correctamente citadas.
- [x] Añadida entrada `ren2015fasterrcnn` (Faster R-CNN) en bib/main.bib.
- [x] Verificada coherencia de todas las citas con las entradas del .bib.

---

## 2. Tareas pendientes (añade las tuyas)

- [ ] Revisar el capítulo de introducción para que mencione correctamente los objetivos actualizados.
- [ ] Revisar si la sección de Contribución Esperada (final de objetivos.tex) sigue siendo coherente con el trabajo real.
- [ ] Añadir el F1-score retroactivamente si se desea (ver sección 3 de este documento).
- [x] ~~Obtener e incorporar diagramas de arquitecturas~~ — Completado: ViT, DINOv3, DEIMv2, ResNet-18, EfficientNet-B0 en fig/diagramas-arquitectura/.
- [ ] Revisar y unificar el estilo de las figuras (mismo tamaño de fuente, leyendas consistentes).
- [ ] Revisión final de ortografía y tipografía (especialmente el uso de comillas españolas «» vs "").
- [ ] Revisar que el resumen (abstract en español e inglés) refleje los resultados finales.
- [ ] Comprobar que todas las referencias en bib/main.bib tienen DOI/URL completos.
- [ ] Revisar la sección de conclusiones del capítulo 4 (¿hay redundancia con el capítulo 6?).

---

## 3. ⚠️ Nota importante sobre F1-score

**Pregunta clave:** ¿Es necesario calcular el F1-score para esta memoria?

### Respuesta directa: No es estrictamente necesario, pero su inclusión fortalecería la alineación con la literatura y la comparabilidad con otros trabajos.

**Por qué no es estrictamente necesario:**
- La métrica AP (Average Precision) integra el tradeoff Precision-Recall sobre todos los umbrales posibles. Es más informativa que un único F1-score a un umbral fijo, porque captura el comportamiento completo del detector.
- mAP@0.5 es el estándar en detección de objetos (COCO, VOC, YOLO papers). Los revisores/tribunal de TFG en el ámbito de detección de objetos reconocen mAP como la métrica principal.

**Por qué conviene añadirlo:**
- El objetivo específico 2 original mencionaba F1-score. Al haberlo eliminado, la coherencia mejora, pero si el tribunal pregunta por F1, conviene tenerlo calculado.
- F1-score = 2·P·R/(P+R). Para DEIMv2, como Precision = 1.0 en todas las clases, el cálculo es trivial: **F1 = 2·Recall / (1 + Recall)**. Puedes calcularlo directamente de las tablas de resultados ya existentes en la memoria.

**F1-scores de DEIMv2 (calculados retroactivamente, 300 épocas, umbral 0.15):**

| Clase | Recall | F1-score |
|-------|--------|----------|
| NORMAL | 0.983 | 0.991 |
| PERFORACIONES | 0.950 | 0.974 |
| DEFORMACIONES | 0.842 | 0.914 |
| CONTAMINACIÓN | 0.788 | 0.882 |
| RAYONES_ARAÑAZOS | 0.853 | 0.920 |
| ROTURA_FRACTURA | 0.725 | 0.841 |
| **Media** | — | **0.920** |

**Para CNNs** (Precision variable): necesitarías ir a los scripts de evaluación y recalcular F1 clase a clase.

**Recomendación:** Añade una columna "F1-score" en las tablas de resultados por clase de DEIMv2 (calculada como se indica arriba). Para CNNs, o bien añades la columna con los valores disponibles en los JSON de evaluación, o simplemente mencionas en el texto que "dada la Precision = 1.0 en todas las clases de DEIMv2, el F1-score equivale a 2·Recall / (1+Recall)". Esto cubre la métrica sin necesidad de nuevas evaluaciones computacionales.

---

## 4. Diagramas de arquitectura recomendados para la memoria

Para un lector académico o tribunal que no esté familiarizado con estas arquitecturas, es muy recomendable incluir diagramas visuales. A continuación se indica dónde buscar recursos de calidad y qué diagrama incluir en cada sección.

### 4.1 Capítulo 3 — Antecedentes y estado del arte

#### ViT (Vision Transformer)
- **Qué mostrar:** División de la imagen en patches → embedding lineal → Transformer Encoder → clasificación por token [CLS].
- **Fuente recomendada:** Figura 1 del artículo original de Dosovitskiy et al. (2021), "An Image is Worth 16x16 Words". Disponible en arXiv:2010.11929.
- **Alternativa libre de derechos:** Reproduce el esquema en TikZ o en draw.io y guárdalo como PDF en `fig/`.

#### DINOv3
- **Qué mostrar:** Arquitectura student-teacher con pérdidas DINO + iBOT + Gram Anchoring.
- **Fuente recomendada:** Figura 2 del paper de Simeoni et al. (2025), "DINOv3: ...". Disponible en arXiv.
- **Alternativa:** Diagrama de bloques con los tres componentes de la función de pérdida.

#### DEIMv2
- **Qué mostrar:** Pipeline completo: imagen → ViT backbone (DINOv3) → STA (dos ramas: semántica y detalle fino) → decoder DETR → predicciones.
- **Fuente recomendada:** Figura principal del paper de Huang et al. (2025), "DEIMv2". Disponible en arXiv.
- **Alternativa:** Diagrama de bloques con los módulos STA + decoder.

### 4.2 Capítulo 4 — Desarrollo

#### CNN clásica (concepto)
- **Qué mostrar:** Esquema genérico Conv → Pool → Conv → Pool → FC. Es suficiente con un diagrama esquemático.
- **Recursos:** Cualquier figura de LeCun et al. o de Deep Learning (Goodfellow), o dibuja en draw.io.

#### ResNet-18
- **Qué mostrar:** Residual block (shortcut connection). Diagrama clásico de un bloque residual básico.
- **Fuente recomendada:** Figura 2 de He et al. (2016), "Deep Residual Learning for Image Recognition". arXiv:1512.03385.
- **Nota:** Este diagrama es tan conocido que re-dibujarlo en TikZ es perfectamente válido académicamente.

#### EfficientNet-B0
- **Qué mostrar:** MBConv block con Squeeze-and-Excitation (SE) + compound scaling (resolución, profundidad, anchura).
- **Fuente recomendada:** Figura 2-3 de Tan & Le (2019), "EfficientNet: Rethinking Model Scaling...". arXiv:1905.11946.

### 4.3 Cómo incluirlos en LaTeX

```latex
\begin{figure}[htbp]
\centering
\includegraphics[width=0.9\textwidth]{fig/diagrama-vit-arquitectura.pdf}
\caption{Arquitectura Vision Transformer (ViT). La imagen se divide en patches de $16\times16$ píxeles que se proyectan a embeddings y se procesan mediante bloques Transformer Encoder. Adaptado de Dosovitskiy et al.~\cite{dosovitskiy2021vit}.}
\label{fig:vit-arquitectura}
\end{figure}
```

Usa el campo `\caption` para indicar "Adaptado de [cita]" si el diagrama es una reproducción o adaptación del original.

---

## 5. Alineación entre capítulo de objetivos y resultados

Se ha realizado una revisión exhaustiva de la alineación entre los objetivos del Capítulo 2 y los resultados del Capítulo 4 y 5. Estado actual:

| Objetivo específico | Estado | Evidencia en la memoria |
|---|---|---|
| OE1: Sistema de detección basado en ViT | ✅ Completado | Sección 4.5: DEIMv2 con DINOv3. mAP=0.785 |
| OE2: Comparativa sistemática con CNNs | ✅ Completado | Fases 1, 3 y tabla resumen (Tab. 4.15). Métricas: mAP, AP, P, R por clase |
| OE3: Dataset industrial curado | ✅ Completado | Sección 4.1: pipeline de curación, 1.022 imágenes, validación estadística |
| OE4: Herramienta de análisis (dashboard) | ✅ Completado | Capítulo 5: dashboard Streamlit con 6 vistas, análisis de umbrales |
| OE5: Validación de robustez | ✅ Completado | Fase 4: umbrales 0.75 y 0.90; Tab. 4.15 con caída controlada del mAP |

**Cabos sueltos detectados — requieren atención:**

1. **F1-score**: Mencionado en el objetivo 2 original (ahora eliminado). Si el tribunal pregunta, ten calculado el F1 de DEIMv2 según la tabla de la sección 3 de este documento.

2. **Visualización de mapas de atención**: El objetivo original de interpretabilidad mencionaba visualización de mecanismos de atención. El dashboard actual muestra predicciones con bounding boxes pero no muestra mapas de atención del ViT. Si el tribunal pregunta, la respuesta honesta es: "La visualización de mapas de atención quedó fuera del alcance final; la interpretabilidad se abordó mediante visualización de predicciones comparativas entre arquitecturas."

3. **Comparativa directa con benchmarks publicados**: La memoria no compara los resultados contra métodos de la literatura en MVTec AD (e.g., ViTAD, UniAD). Esto es aceptable dado que el dataset es un subset curado unificado, no el MVTec original completo. Conviene añadir una frase en las conclusiones que lo justifique explícitamente.

4. **Tiempo de inferencia**: Mencionado como objetivo pragmático (ahora simplificado). No hay datos de latencia en la memoria. Conviene añadir una nota breve con los fps aproximados de DEIMv2 en la GPU disponible.

---

## 6. Pequeños remates finales recomendados

Estos son detalles que, aunque menores, marcan la diferencia entre un TFG notable y uno sobresaliente:

- **Numeración consistente de figuras y tablas**: Revisa que ninguna figura/tabla referencie un número incorrecto en el texto (especialmente tras añadir nuevas secciones).
- **Comillas**: En español académico se usan «comillas angulares» o "comillas inglesas" con consistencia. Evitar mezclas.
- **Abreviatura vs. acrónimo**: Asegúrate de que la primera vez que aparece un acrónimo en el texto se usa `\acrfull{}` y las siguientes `\acrshort{}`.
- **Unidades consistentes**: Revisar que todas las resoluciones se expresan igual (1024×1024 o $1024\times1024$).
- **Conclusiones vs. resultados**: Verificar que el capítulo 5 (Resultados) y el capítulo 6 (Conclusiones) no se solapan en contenido; el capítulo 5 debe ser descriptivo/analítico y el 6 debe ser sintético/prospectivo.
- **Futuras líneas bien justificadas**: En la sección 6.2, las líneas de futuro deben estar ordenadas de mayor a menor impacto potencial. Considera añadir una estimación del esfuerzo computacional (e.g., "requeriría GPU A100 o superior") para dar contexto al tribunal.
- **Revisión del resumen/abstract**: El abstract es lo primero que lee el tribunal. Asegúrate de que menciona: el problema (detección de anomalías industriales), el enfoque (ViT + DEIMv2 + DINOv3), el resultado principal (mAP 0.785 vs 0.162 mejor CNN), y la contribución (pipeline de curación + herramienta de análisis).
