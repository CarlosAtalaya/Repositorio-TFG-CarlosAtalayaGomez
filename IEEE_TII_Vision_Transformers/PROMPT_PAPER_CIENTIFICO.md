# PROMPT: Redacción de Paper Científico basado en TFG

## 🎯 OBJETIVO

Necesito ayuda para redactar un **paper científico de investigación** basado en mi Trabajo Fin de Grado (TFG) sobre **Vision Transformers para Detección de Defectos Industriales**. El paper debe seguir la plantilla que compartiré a continuación y sintetizar los hallazgos más relevantes de la memoria técnica.

---

## 📚 DOCUMENTACIÓN DE REFERENCIA (MEMORIA TFG)

La memoria técnica completa está estructurada en los siguientes archivos LaTeX. **Debes leerlos para extraer el contenido científico**:

### Archivos principales (OBLIGATORIO leer):

| Archivo | Contenido | Propósito para el paper |
|---------|-----------|------------------------|
| `@tex/introduccion.tex` | Contexto del problema, motivación, ViT vs CNN, SSL | **Introduction** del paper |
| `@tex/antecedentes.tex` | Estado del arte exhaustivo (ViT, SSL, DINOv2, detección anomalías) | **Related Work** del paper |
| `@tex/objetivos.tex` | Objetivos, alcance, hipótesis, contribución esperada | Definir **scope** y **contributions** |
| `@tex/desarrollo.tex` | Metodología, experimentos (Fases 0-3), resultados, análisis | **Methodology**, **Experiments**, **Results** |

### Estructura del capítulo de desarrollo (resumen ejecutivo):

- **Fase 0**: Curación de dataset (VISION-Datasets + MVTec AD → 1,022 imágenes, 6 categorías unificadas)
- **Fase 1**: Baseline CNN (ResNet-18: mAP=0.077, EfficientNet-B0: mAP=0.162)
- **Fase 2**: DEIMv2 (Vision Transformer) con 4 iteraciones → **mAP=0.785** (mejor modelo)
- **Fase 3**: Validación experimental (CNNs a 1024×1024 no mejoran → superioridad arquitectónica de ViT confirmada)

---

## 🔬 HALLAZGOS CLAVE (para Abstract y Conclusions)

1. **DEIMv2 supera a CNNs por +881%** (mAP 0.785 vs 0.077-0.162)
2. **Resolución crítica para ViTs**: 1024×1024 aporta +25% mAP vs 640×640
3. **Convergencia lenta pero robusta**: ViTs requieren 150-200 épocas vs 50 de CNNs
4. **Precision perfecta (1.0)**: DEIMv2 no genera falsos positivos
5. **CNNs no aprovechan alta resolución**: EfficientNet empeora -24.7% a 1024px

---

## 📐 RESTRICCIONES DE REDACCIÓN

### Estilo académico:
- **Idioma**: Español de España (formal, impersonal o plural mayestático)
- **Voz**: "Se ha realizado...", "Los resultados muestran...", "Hemos observado..."
- **Evidencia**: Toda afirmación debe estar respaldada por datos de los experimentos
- **Citas**: Usar formato de la plantilla (se proporcionará)

### Estructura esperada del paper:
1. **Title** (conciso, descriptivo)
2. **Abstract** (150-250 palabras)
3. **Introduction** (problema, motivación, contribución)
4. **Related Work** (estado del arte sintetizado)
5. **Methodology** (arquitectura, dataset, configuración experimental)
6. **Experiments & Results** (tablas comparativas, análisis)
7. **Discussion** (interpretación, limitaciones)
8. **Conclusions & Future Work**
9. **References**

---

## 📄 PLANTILLA DEL PAPER

**[INSERTAR AQUÍ LA RUTA A LA PLANTILLA]**

Ejemplo: `@ruta/a/plantilla/paper_template.tex`

---

## ✅ INSTRUCCIONES PARA EL AGENTE

1. **Lee primero** los 4 archivos de referencia para comprender el contexto completo
2. **Sintetiza** el contenido para un paper (no copies literalmente, adapta al formato)
3. **Prioriza** los resultados cuantitativos y las conclusiones más impactantes
4. **Mantén** coherencia con la terminología y nomenclatura de la memoria
5. **Adapta** al formato y estilo de la plantilla proporcionada
6. **Genera** el contenido sección por sección, permitiendo revisión iterativa

---

## 📊 DATOS CLAVE PARA TABLAS DEL PAPER

### Tabla comparativa principal (mAP@0.5):

| Arquitectura | Resolución | Épocas | mAP@0.5 |
|--------------|------------|--------|---------|
| ResNet-18 | Nativa | 50 | 0.077 |
| ResNet-18 | 1024×1024 | 50 | 0.080 |
| EfficientNet-B0 | Nativa | 50 | 0.162 |
| EfficientNet-B0 | 1024×1024 | 50 | 0.122 |
| DEIMv2 | 640×640 | 87 | 0.499 |
| DEIMv2 | 1024×1024 | 80 | 0.624 |
| DEIMv2 | 1024×1024 | 120 | 0.766 |
| **DEIMv2** | **1024×1024** | **300** | **0.785** |

### Dataset final:
- **Imágenes**: 1,022 (train: 715, val: 102, test: 205)
- **Categorías**: 6 (NORMAL, DEFORMACIONES, ROTURA_FRACTURA, RAYONES_ARAÑAZOS, PERFORACIONES, CONTAMINACIÓN)
- **Fuentes**: VISION-Datasets + MVTec AD
- **Balance**: Ratio máx/mín = 2.08:1

---

## 🚀 CÓMO EMPEZAR

1. Lee `@tex/introduccion.tex` y `@tex/antecedentes.tex` para el marco teórico
2. Lee `@tex/objetivos.tex` para entender el alcance y contribución
3. Lee `@tex/desarrollo.tex` para los experimentos y resultados
4. Revisa la plantilla del paper que proporcionaré
5. Comienza por el **Abstract** (síntesis de todo el trabajo)
6. Continúa sección por sección según la estructura de la plantilla

---

*Documento generado para facilitar la redacción del paper científico derivado del TFG "Vision Transformers para Detección de Defectos Industriales" - Diciembre 2025*

