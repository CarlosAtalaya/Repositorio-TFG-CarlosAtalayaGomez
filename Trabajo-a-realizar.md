# 📘 HOJA DE RUTA: PULIDO INTEGRAL DE LA MEMORIA TÉCNICA
**Proyecto:** Detección de Anomalías Industriales mediante Vision Transformers y Self-Supervised Learning.
**Institución:** Escuela Superior de Ingeniería Informática (UCLM).
**Referencia Principal:** `CONTEXTUALIZACION-REPOSITORIO.txt`

---

## ESTADO ACTUAL (Última actualización)
| Fase | Estado | Observaciones |
|------|--------|---------------|
| 1. Personalización e Identidad | ✅ Completada | Metadata, resumen, estructura, agradecimientos, DINOv3 |
| 2. Cap. 3 Antecedentes | ✅ Completada | Ecuaciones Self-Attention, tabla CNN vs ViT, transiciones |
| 3. Cap. 4 Desarrollo | ✅ Completada | Fase 4 robustez, narrativa 82%, punto óptimo, experimento crucial |
| 4. Cap. 5 Resultados | 🔄 Siguiente | Pendiente: dashboard, Fase 4 detallada |
| 5. Cap. 6 Conclusiones | ⏳ Pendiente | |
| 6. Auditoría Final | ⏳ Pendiente | |

---

## 0. PREMISAS DE CALIDAD Y ESTILO (ESTRICTO)
*Estas reglas rigen toda la redacción para cumplir con las normas de la UCLM y las directrices internas del repositorio.*

1.  **Voz Académica:** Uso exclusivo de la **pasiva refleja** ("Se ha analizado") o el **plural mayestático** ("Hemos validado"). Queda prohibida la primera persona del singular.
2.  **Sintaxis LaTeX:** El contenido final debe integrarse directamente en los archivos `.tex` sin bloques de código Markdown.
3.  **Gestión de Citas:** Usar únicamente las claves de `bib/main.bib`. **No inventar citas**.
4.  **Flujo Narrativo:** Cada capítulo debe cerrar con una frase de transición que anticipe el siguiente para garantizar la cohesión del "hilo conductor".

---

## 1. FASE DE PERSONALIZACIÓN E IDENTIDAD ✅ COMPLETADA
**Objetivo:** Eliminar la sensación de "plantilla genérica" y adaptar la estructura a los TFGs de excelencia (Lapeña, 2014; Martínez, 2019).

- [x] **Acción 1:** Revisar `TFG.tex` y `datos-tfg.tex` para asegurar que los metadatos y la jerarquía de capítulos reflejan una investigación original.
- [x] **Acción 2:** Redactar el `resumen.tex` (Abstract) con enfoque periodístico-científico.
- [x] Auditoría DINOv2 vs DINOv3 aplicada en introducción, objetivos, antecedentes, resumen y desarrollo.
- [x] Eliminación de restos de plantilla (avisoLocalizacionArchivo, Chuleta LaTeX, citas de ejemplo).
- [x] Agradecimientos y dedicatoria personalizados.

---

## 2. CAPÍTULO 3: ANTECEDENTES (INTENSIFICACIÓN TÉCNICA) ✅ COMPLETADA
**Objetivo:** Reflejar una cultura ingenieril profunda y corregir inconsistencias bibliográficas.

- [x] **Auditoría DINOv2 vs DINOv3:** Ya aplicada. DINOv3 como backbone propio; DINOv2 conservado en citas de trabajo de otros autores.
- [x] **Rigor Matemático:** Ecuaciones de Self-Attention insertadas en `tex/antecedentes.tex`.
- [x] **Tablas Científicas:** Tabla comparativa CNN vs ViT con booktabs.

---

## 3. CAPÍTULO 4: DESARROLLO (EL CORAZÓN DE LA INVESTIGACIÓN) ✅ COMPLETADA
**Objetivo:** Sincronizar el contenido, identificar lagunas y elevar la calidad del relato para un tribunal académico.

### 3.1. Sincronización y Auditoría de Contenidos
- [x] Comparativa crítica realizada entre `tex/desarrollo.tex` y `FASE_EXPERIMENTACION.md`.
- [x] Fase 4 (Validación de robustez) integrada con metodología y tabla de umbrales 0.75/0.90.
- [x] Métricas de Fases 3 y 4 sincronizadas.

### 3.2. Calidad de Redacción y Storytelling
- [x] **Justificación de la Resolución:** 82% de pérdida de información a 640px, invalidando defectos milimétricos; 47% preservado a 1024px.
- [x] **Narrativa de Fases:** Fase 1 como motor de búsqueda; Fase 2 con punto óptimo (época 187); Fase 3 como experimento crucial.

---

## 4. CAPÍTULO 5: RESULTADOS Y HERRAMIENTA DE ANÁLISIS
**Objetivo:** Consolidar la validación científica y presentar la implementación de software adicional.

1.  **Diferenciación Técnica:** El agente debe separar claramente la *narrativa de la experimentación* (Cap. 4) de la *discusión de resultados finales* (Cap. 5).
2.  **Integración del Dashboard (Flask):**
    * Redactar una sección descriptiva sobre `dashboard.py`.
    * **Telling Story:** Explicar que la herramienta nace para la **interpretabilidad**. Las métricas numéricas no explicaban la confusión entre ROTURA y RAYONES; el dashboard permitió visualizar mapas de atención y confirmar que el ViT capturaba la discontinuidad estructural del material.
3.  **Validación de Robustez (Fase 4):** Incluir el análisis con thresholds de 0.75 y 0.90 para demostrar que el modelo no sufre de *overfitting* y mantiene una precisión del 1.0 (100%) incluso en condiciones estrictas.

---

## 5. CAPÍTULO 6: CONCLUSIONES Y TRABAJOS FUTUROS
**Objetivo:** Reflexión crítica y honestidad científica sobre la multimodalidad.

1.  **La "Paradoja Multimodal":** Utilizar `why-fase2.txt` y `FASE2_Estrategias_Implementacion.md` para explicar por qué la inyección de texto (CLIP) no mejoró el mAP de 0.785.
    * **Argumento:** El modelo visual (DINOv3) ya es un "experto" tan optimizado que la señal textual añadía ruido. Esto debe presentarse como un **resultado negativo de gran valor científico**.
2.  **Estrategia "Zero-Start":** Describir cómo se intentó mitigar el olvido catastrófico mediante el descongelamiento progresivo del backbone.

---

## 6. AUDITORÍA FINAL DE FORMATO (CHECKLIST)
* [ ] **Figuras:** Verificar que todas las gráficas de pérdida y métricas de `Documentacion-externa/` están correctamente referenciadas con `\begin{figure}`, `\caption` y `\label`.
* [ ] **Tablas:** Usar `booktabs` (`\toprule`, `\midrule`, `\bottomrule`) para una estética profesional.
* [ ] **Acrónimos:** Asegurar que todos los términos (ViT, SSL, mAP, IoU, Flask) están definidos en `tex/acronimos.tex`.
* [ ] **Rutas:** Ajustar rutas de imágenes para que apunten correctamente a `fig/` o subcarpetas externas.

---
**Nota para el Agente Redactor:** Tu misión es convertir los datos fríos de los archivos `.md`, `.json` y `.py` en una narrativa de ingeniería de alta gama, justificando cada decisión técnica con la evidencia recolectada.