# 📘 HOJA DE RUTA: PULIDO INTEGRAL DE LA MEMORIA TÉCNICA
**Proyecto:** Detección de Anomalías Industriales mediante Vision Transformers y Self-Supervised Learning.
**Institución:** Escuela Superior de Ingeniería Informática (UCLM).
**Referencia Principal:** `CONTEXTUALIZACION-REPOSITORIO.txt`

---

## ESTADO ACTUAL (Última actualización)

### ✅ COMPLETADO RECIENTEMENTE
| Tarea | Estado | Detalle |
|-------|--------|---------|
| Cap. 5 Resultados | ✅ Completado | Integración dashboard Streamlit, evidencia visual, figuras y predicciones |
| Formato figuras/tablas desarrollo.tex | ✅ Completado | Pies de tabla debajo (\caption tras \end{tabular}); \FloatBarrier para control de flotantes |
| Tabla 4.1 | ✅ Completado | Recolocada con [H] justo tras referencia en «Metodología de curación», antes de Etapa 1 |
| Diseño y tecnología (resultados.tex) | ✅ Completado | Lista itemize con espaciado correcto |

### 🔄 PENDIENTE DE REVISIÓN
| Fase | Estado | Observaciones |
|------|--------|---------------|
| 1. Personalización e Identidad | ✅ Completada | Metadata, resumen, agradecimientos |
| 2. Cap. 3 Antecedentes | 🔄 Correcciones | Título 3.2 y Tabla 3.1 fuera de márgenes; más detalle ViT/DINOv3/DEIMv2 |
| 3. Cap. 4 Desarrollo | 🔄 Justificaciones | Parámetros, score th 0.15, precision 1.0, métricas, batch size |
| 4. Cap. 5 Resultados | ✅ Completado | Dashboard, evidencia visual |
| 5. Cap. 6 Conclusiones | 🔄 Reestructurar | Separar Conclusiones y Trabajos a futuro |
| 6. Formato y maquetación | ⏳ Pendiente | Numeración, márgenes, cabeceras, acrónimos |

---

## 0. PREMISAS DE CALIDAD Y ESTILO (ESTRICTO)
*Estas reglas rigen toda la redacción para cumplir con las normas de la UCLM y las directrices internas del repositorio.*

1.  **Voz Académica:** Uso exclusivo de la **pasiva refleja** ("Se ha analizado") o el **plural mayestático** ("Hemos validado"). Queda prohibida la primera persona del singular.
2.  **Sintaxis LaTeX:** El contenido final debe integrarse directamente en los archivos `.tex` sin bloques de código Markdown.
3.  **Gestión de Citas:** Usar únicamente las claves de `bib/main.bib`. **No inventar citas**.
4.  **Flujo Narrativo:** Cada capítulo debe cerrar con una frase de transición que anticipe el siguiente para garantizar la cohesión del "hilo conductor".

---

## 1. FORMATO Y MAQUETACIÓN (PRIORITARIO)

### 1.1. Listado de siglas o acrónimos
- [ ] Incluir el listado completo de siglas/acrónimos en la memoria.
- [ ] Verificar que `tex/acronimos.tex` cubra todos los términos usados (ViT, CNN, mAP, IoU, SSL, DINOv3, DEIMv2, AP, RPN, etc.) y que el índice de acrónimos se genere correctamente.

### 1.2. Numeración de páginas
- [ ] **Problema:** La página 1 está en parte inferior central; la 2 en superior izquierda; la 3 en superior derecha.
- [ ] **Objetivo:** Todas las numeraciones de página deben ir en la **parte superior derecha**.
- [ ] Revisar `sty/eiiatfg.cls` y paquetes de encabezado/pie (fancyhdr, pagestyle) para unificar la posición.

### 1.3. Márgenes
- [ ] **Problema:** Páginas pares parecen centradas; páginas impares desplazadas ligeramente a la derecha con márgenes diferentes.
- [ ] Revisar `geometry` y configuración de páginas impares/pares en la clase para garantizar simetría y centrado.

### 1.4. Cabeceras de página dentro de capítulos
- [ ] **Problema:** Tras subsecciones (p. ej. 2.2.3. Validación en datasets benchmark estándar), al hacer salto de página, la cabecera muestra «2.2. OBJETIVOS ESPECÍFICOS» en lugar de «CAPÍTULO 2. OBJETIVOS». Este error es recurrente y parece afectar solo a **páginas impares**.
- [ ] Revisar la lógica de cabeceras (headings) en `sty/eiiatfg.cls` o `indices.sty` para que la cabecera superior refleje correctamente el capítulo actual.

---

## 2. CAPÍTULO 3: ANTECEDENTES — CORRECCIONES

### 2.1. Errores de formato
- [ ] **Sección 3.2:** El título «3.2. Fundamentos de Vision Transformers y aprendizaje auto-supervisado» no respeta los márgenes y se sale hacia la derecha.
- [ ] **Tabla 3.1:** Se sale mucho del margen hacia la derecha. Ajustar anchura o diseño.

### 2.2. Tablas con pie descriptivo
- [ ] Revisar que **todas** las tablas tengan descripción y referencia en el pie (caption).
- [ ] Ejemplo: Tabla 3.1 debe tener un pie como «Comparativa arquitectural entre CNN y Vision Transformer.».

### 2.3. DEIMv2 en sección 3.8
- [ ] Confirmar que la arquitectura DEIMv2 esté **introducida y descrita** en la sección 3.8 de Antecedentes.
- [ ] En el capítulo de Desarrollo **no** debe describirse DEIMv2; solo usarse como arquitectura ya presentada.

### 2.4. Profundidad técnico-científica (ViT, DINOv3, DEIMv2)
- [ ] Añadir más detalle técnico-científico sin ser excesivamente cargante.
- [ ] Incluir al menos **un diagrama representativo** de DEIMv2 (el paper original tiene un diagrama muy representativo; reciclar o adaptar).

---

## 3. CAPÍTULO 4: DESARROLLO — JUSTIFICACIONES

### 3.1. Contexto técnico de métricas (antes de experimentos)
- [ ] En la subsección «Métricas de evaluación», dar al lector un contexto técnico claro de:
  - **mAP@0.5**
  - **AP por clase**
  - **Precision por clase**
  - **Recall por clase**
- [ ] Insertar **fórmulas matemáticas** si aportan valor para definir estas métricas.

### 3.2. Justificación de parámetros de entrenamiento
- [ ] Siempre que se presenten parámetros de un experimento, **justificar** por qué son óptimos para ese tipo de arquitectura.
- [ ] Incluir explicación de qué es cada parámetro: p. ej. **Batch Size**, Learning Rate, Optimizador, etc. El lector debe comprender su función.

### 3.3. Justificación de la evaluación en test
- [ ] Explicar que durante la fase de experimentación, para validar si los experimentos son exitosos, se **evalúa cada modelo contra el subconjunto de test** en un mismo entorno controlado.
- [ ] Justificar que la métrica más informativa es el **AP por clase**, para ver el comportamiento del modelo por categoría.

### 3.4. Justificación del score threshold 0.15
- [ ] Explicar que el score threshold utilizado para la evaluación se estableció **arbitrariamente en 0.15**.
- [ ] **Motivo:** Con las redes tradicionales (CNNs) no se conseguían muchas detecciones; las que daban lo hacían con un índice de confianza muy bajo. Se estableció 0.15 para ser más justos con las redes convencionales en la comparativa.

### 3.5. Justificación MUY DETALLADA de precision 1.0 en DEIMv2
- [ ] Hacer **énfasis** en que los valores de precision 1.0 para todos los experimentos con DEIMv2 **no son ficticios ni indican error**.
- [ ] Explicar: la arquitectura DEIMv2 obtiene un número de detecciones mucho mayor; al ser capaz de obtener siempre alguna detección por encima del umbral 0.15 que coincide con la etiqueta real, la precision resulta del 100%. Justificar correctamente este fenómeno para evitar dudas del tribunal.

---

## 4. CAPÍTULO 5: RESULTADOS
**Estado:** ✅ Completado (dashboard Streamlit, evidencia visual, comparativas, predicciones).

---

## 5. CAPÍTULO 6: REESTRUCTURACIÓN — CONCLUSIONES Y TRABAJOS A FUTURO

### 5.1. Estructura nueva
- [ ] **Separar** en dos bloques independientes:
  1. **Sección de Conclusiones**
  2. **Capítulo o sección de Trabajos a Futuro**

### 5.2. Contenido de Conclusiones
- [ ] Reflejar muy bien el trabajo realizado.
- [ ] Poner en valor la **unificación de datasets** para obtener un conjunto único con cierta variedad de datos.
- [ ] Hablar de los resultados obtenidos.
- [ ] Explicar por qué **DEIMv2 con arquitectura ViT es tan potente**.
- [ ] Hablar del **comportamiento del modelo en situaciones adversas** (entornos menos controlados).

### 5.3. Contenido de Trabajos a Futuro
- [ ] Contextualizar correctamente:
  - `Documentacion-externa/FASE-2-analisis/why-fase2.txt`
  - `Documentacion-externa/FASE-2-analisis/FASE2_Estrategias_Implementacion.md`
  - `Documentacion-externa/FASE-2-analisis/fase2-entrenamiento-progresivo/fase2-implementacion-entrenamientoprogresivo-conlcuisones.md`
- [ ] Mencionar que se podría compartir contenido gráfico y métricas en:
  - `Documentacion-externa/FASE-2-analisis/fase2-entrenamiento-progresivo/cosas-repo-codigo/analysis_fase2_entrenamiento_progresivo/`
- [ ] Reconocer que no se ha hecho una exploración muy exhaustiva en estas mejoras del ViT.
- [ ] **Foco principal:** Dejar énfasis en explorar la vía de **VLMs (Vision-Language Models)**.
- [ ] Existen variantes de código abierto muy potentes como **QWEN3-VL**, que quizá consuman más recursos que un ViT como DINOv3, pero la **multimodalidad** y la **comprensión** que adquieren son muy elevadas.
- [ ] Proponer QWEN3-VL como línea de investigación futura que podría mejorar los trabajos realizados.
- [ ] El autor buscará las referencias bibliográficas de QWEN3-VL; el agente debe sintetizar la información correctamente cuando estén disponibles.

---

## 6. AUDITORÍA FINAL (CHECKLIST)
- [ ] Figuras: todas con `\begin{figure}`, `\caption`, `\label`.
- [ ] Tablas: booktabs; caption y label en pie de tabla.
- [ ] Acrónimos: listado completo y coherente.
- [ ] Rutas de imágenes: correctas (`fig/` o rutas externas).
- [ ] Comparativas: usar términos absolutos (pp) en lugar de porcentajes relativos cuando convenga.

---

## 7. REFERENCIAS DOCUMENTALES PARA EL AGENTE
- **Contexto global:** `CONTEXTUALIZACION-REPOSITORIO.txt`
- **Memoria TFG:** `tex/desarrollo.tex`, `tex/resultados.tex`, `tex/antecedentes.tex`, `tex/conclusiones.tex`
- **Fase 2:** `why-fase2.txt`, `FASE2_Estrategias_Implementacion.md`, `fase2-implementacion-entrenamientoprogresivo-conlcuisones.md`
- **Bibliografía:** `bib/main.bib` (solo claves existentes)
- **Reglas:** `.cursorrules`, clase `sty/eiiatfg.cls`

---

**Nota para el Agente Redactor:** La misión es convertir los datos y documentación en una narrativa de ingeniería de alta calidad, justificando cada decisión técnica con evidencia. Antes de redactar, consultar `notas-correcciones-memoria.txt` (sección «PROMPT PARA EL PRÓXIMO AGENTE») y `CONTEXTUALIZACION-REPOSITORIO.txt`.
