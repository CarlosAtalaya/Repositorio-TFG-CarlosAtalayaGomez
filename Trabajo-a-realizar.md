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
| 3. Cap. 4 Desarrollo | 🔄 Pendiente revision | Comparativas: usar términos absolutos (p. ej. pp), no porcentajes relativos |
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

## 3. CAPÍTULO 4: DESARROLLO (EL CORAZÓN DE LA INVESTIGACIÓN)
**Objetivo:** Sincronizar el contenido, identificar lagunas y elevar la calidad del relato para un tribunal académico.

### 3.1. Sincronización y Auditoría de Contenidos
- [x] Comparativa crítica realizada entre `tex/desarrollo.tex` y `FASE_EXPERIMENTACION.md`.
- [x] Fase 4 (Validación de robustez) integrada con metodología y tabla de umbrales 0.75/0.90.
- [x] Métricas de Fases 3 y 4 sincronizadas.

### 3.2. Calidad de Redacción y Storytelling
- [x] **Justificación de la Resolución:** 82% de pérdida de información a 640px, invalidando defectos milimétricos; 47% preservado a 1024px.
- [x] **Narrativa de Fases:** Fase 1 como motor de búsqueda; Fase 2 con punto óptimo (época 187); Fase 3 como experimento crucial.

### 3.3. Comparativas entre arquitecturas: términos absolutos (pendiente)
- [ ] **Evitar porcentajes relativos** en las comparaciones de modelos/arquitecturas. En el mundo académico no resulta apropiado expresar diferencias como «X es 500% mejor que Y» cuando, por ejemplo, una arquitectura tiene mAP 0.15 y otra 0.75.
- [ ] **Usar siempre términos absolutos:** indicar cuántos puntos porcentuales (pp) o puntos proporcionales está una arquitectura por encima o por debajo de otra (p. ej. «DEIMv2 supera a ResNet-18 en 60 puntos porcentuales» o «0.60 puntos proporcionales de mAP»).

---

## 4. CAPÍTULO 5: RESULTADOS Y HERRAMIENTA DE ANÁLISIS
**Objetivo:** Consolidar la validación científica, presentar la herramienta de análisis Streamlit y el material visual obtenido desde el dashboard.

### 4.0. SÍNTESIS DE CONTEXTO Y AUTONOMÍA INTERPRETATIVA
**Situación:** El Capítulo 4 (`tex/desarrollo.tex`) ya documenta exhaustivamente los resultados cuantitativos: tablas de mAP, AP por clase, robustez (Fase 4), resumen de experimentos, curvas de entrenamiento. Existe un solapamiento natural con lo que podría incluirse en Cap. 5. El agente debe **sintetizar el contexto** y **interpretar** con criterio experto cómo estructurar y jerarquizar ambos capítulos para alcanzar un índice de coherencia muy alto y que la narrativa no se pierda.

**Recursos que el agente debe sintetizar:**
- **Memoria del proyecto:** `tex/desarrollo.tex` íntegro (inventario de tablas, figuras, hilo narrativo).
- **Memorias TFG de referencia:** `Documentacion-externa/Memorias-tfg-ejemplos/` (TFG_JavierLapeñaParreño.pdf, TFG_BernaroMartínezParras.pdf). Extraer patrones de cómo TFGs de excelencia separan Desarrollo vs Resultados.
- **Material visual:** `fig/` (evidencia de entrenamiento; subcarpetas `resultados_dashboard/` y `resultados_predicciones/` para gráficas y pantallazos del dashboard).

**Objetivo:** El agente, actuando como experto en la materia y en redacción académica, determinará la estructura y jerarquía óptimas de los apartados de desarrollo.tex y resultados.tex. La diferenciación entre ambos capítulos debe ser clara, el hilo narrativo coherente y la calidad académica excelsa.

### 4.1. Diferenciación técnica
- Separar la *narrativa de la experimentación* (Cap. 4) de la *discusión de resultados finales* (Cap. 5).
- Abrir el capítulo con referencia explícita al Cap. 4 para mantener el hilo narrativo.
- Las métricas (mAP 0.785, etc.) se tratarán en forma de síntesis con referencia al Cap. 4 para el detalle (interpretación del agente).

### 4.2. Integración del Dashboard Streamlit
- **Tecnología:** Dashboard basado en **Streamlit** (no Flask). Ubicación: `Documentacion-externa/FASE-1-analisis/herramienta_comparativa/dashboard.py`. Uso: `streamlit run dashboard.py` desde la carpeta herramienta_comparativa.
- **Telling Story:** La herramienta nace para la **interpretabilidad**. Las métricas numéricas no explicaban la confusión entre ROTURA_FRACTURA y RAYONES_ARAÑAZOS; el dashboard permitió visualizar predicciones y confirmar que el ViT captura mejor la discontinuidad estructural del material.
- Describir las vistas clave: **Explorador** (gráficos AP/Precision/Recall, métricas de entrenamiento, análisis de thresholds), **Comparativa** (mAP global, AP por clase), **Visualizaciones** (Ground Truth vs predicciones por arquitectura).

### 4.3. Material visual desde el dashboard
- **Gráficas:** `fig/resultados_dashboard/`. El agente evaluará qué gráficas aportan valor aditivo respecto a desarrollo.tex (véase 4.0).
- **Pantallazos de predicciones:** `fig/resultados_predicciones/`. Capturas de la vista **Visualizaciones** (Ground Truth vs ResNet-18 vs EfficientNet-B0 vs DEIMv2). Este material no existe en Cap. 4 y constituye valor visual complementario.

### 4.4. Validación de robustez (Fase 4)
- La tabla `tab:fase4_robustez` está en desarrollo.tex. El agente interpretará cómo tratar este contenido en Resultados sin duplicar (p. ej. discusión que referencie Cap. 4 y sintetice el mensaje clave).

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
* [ ] **Acrónimos:** Asegurar que todos los términos (ViT, SSL, mAP, IoU, Streamlit) están definidos en `tex/acronimos.tex`.
* [ ] **Rutas:** Ajustar rutas de imágenes para que apunten correctamente a `fig/` o subcarpetas externas.

---
**Nota para el Agente Redactor:** Tu misión es convertir los datos fríos de los archivos `.md`, `.json` y `.py` en una narrativa de ingeniería de alta gama, justificando cada decisión técnica con la evidencia recolectada.

**Antes de redactar el Cap. 5:** Consultar `notas-correcciones-memoria.txt` (sección «PROMPT PARA EL PRÓXIMO AGENTE»). El prompt define una **tarea de síntesis previa**: integrar la memoria del proyecto (desarrollo.tex), las memorias TFG de referencia (Documentacion-externa/Memorias-tfg-ejemplos/) y el material visual (fig/). A partir de esa síntesis, interpretarás con criterio experto cómo estructurar y jerarquizar desarrollo.tex y resultados.tex para maximizar la coherencia y el hilo narrativo.