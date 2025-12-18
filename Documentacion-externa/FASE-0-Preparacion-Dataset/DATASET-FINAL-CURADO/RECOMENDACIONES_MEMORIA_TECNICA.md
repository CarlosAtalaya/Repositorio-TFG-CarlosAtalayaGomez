# Recomendaciones para la Memoria Técnica: Documentación del Proceso de Curación del Dataset

## Introducción

Este documento presenta mis recomendaciones sobre qué contenido gráfico, métricas y elementos visuales incluir en la memoria técnica de tu proyecto para documentar de manera rigurosa y científica el proceso de curación del dataset para detección de defectos industriales.

La preparación de datos representa típicamente el **60-80% del esfuerzo** en proyectos de Machine Learning, y su correcta documentación es fundamental para:

1. **Reproducibilidad científica** del proceso
2. **Trazabilidad** de decisiones de diseño
3. **Validación** de la calidad del dataset
4. **Comunicación** del trabajo realizado

---

## Estructura Recomendada para la Memoria Técnica

### Sección 1: Análisis de Datasets Originales

#### Contenido Textual
- Descripción de VISION-Datasets y MVTec AD
- Características técnicas de cada dataset (formato, resolución, tipos de defectos)
- Justificación de la selección de estos datasets

#### Material Gráfico Recomendado

| Elemento | Ubicación | Propósito |
|----------|-----------|-----------|
| **Tabla comparativa** de datasets originales | `datasets_technical_summary.txt` | Mostrar diferencias entre VISION y MVTec |
| **Diagrama de categorías** por dataset | `detailed_categories_analysis.txt` | Visualizar la diversidad de defectos |
| **Muestras de imágenes** representativas | Datasets originales | Ilustrar la variabilidad visual |

**Figura sugerida 1:** *"Taxonomía de defectos en datasets originales"*
- Esquema jerárquico mostrando VISION-Datasets (14 componentes, 44 tipos de defectos) vs MVTec AD (15 categorías, 49 tipos de defectos)

**Figura sugerida 2:** *"Mosaico de ejemplos por tipo de defecto"*
- Grid 4×6 con ejemplos de cada tipo principal de defecto de ambos datasets

---

### Sección 2: Diseño de Taxonomía Unificada

#### Contenido Textual
- Criterios de unificación de categorías
- Mapeo semántico entre defectos de ambos datasets
- Justificación de la taxonomía de 6 categorías

#### Material Gráfico Recomendado

**Figura sugerida 3:** *"Diagrama de mapeo de categorías"*

```
┌─────────────────────────────────────────────────────────────────┐
│                    TAXONOMÍA UNIFICADA                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  VISION-Datasets          CATEGORÍA FINAL          MVTec AD     │
│  ────────────────         ─────────────           ──────────    │
│                                                                  │
│  good, normal       ──►   NORMAL           ◄──    good          │
│                                                                  │
│  short, spur        ──►   DEFORMACIONES    ◄──    bent, bent_*  │
│                                                                  │
│  break, defect      ──►   ROTURA_FRACTURA  ◄──    crack, broken │
│                                                                  │
│  Scratch, s_scratch ──►   RAYONES_ARANAZOS ◄──    scratch       │
│                                                                  │
│  Hole, missing_hole ──►   PERFORACIONES    ◄──    hole, cut     │
│                                                                  │
│  Dirty              ──►   CONTAMINACION    ◄──    contamination │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

**Tabla sugerida:** *"Correspondencia semántica de defectos"*
- Ubicación: `category_mapping.csv` de Etapa 4.2

---

### Sección 3: Pipeline de Curación

#### Contenido Textual
- Descripción de cada etapa del pipeline
- Justificación técnica de cada operación
- Métricas de reducción/transformación

#### Material Gráfico Recomendado

**Figura sugerida 4:** *"Diagrama de flujo del pipeline de curación"*

```
┌────────────────────────────────────────────────────────────────────────┐
│                      PIPELINE DE CURACIÓN                              │
├────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   ┌───────────────┐     ┌───────────────┐     ┌───────────────┐        │
│   │ VISION-Data   │     │   ETAPA 1     │     │   ETAPA 2     │        │
│   │ (3,788 imgs)  │────►│ Exploración   │────►│   Curación    │        │
│   └───────────────┘     │ + Análisis    │     │   Inicial     │        │
│                         └───────────────┘     └───────────────┘        │
│   ┌───────────────┐            │                     │                  │
│   │   MVTec AD    │            │                     │                  │
│   │ (5,354 imgs)  │────────────┘                     │                  │
│   └───────────────┘                                  ▼                  │
│                                              ┌───────────────┐          │
│                                              │   1,907 imgs  │          │
│                                              │   15 cats     │          │
│                                              └───────┬───────┘          │
│                                                      │                  │
│                                                      ▼                  │
│   ┌───────────────────────────────────────────────────────────────┐    │
│   │                       ETAPA 4 - RECURACIÓN                     │    │
│   ├───────────────────────────────────────────────────────────────┤    │
│   │                                                                │    │
│   │  4.1 Re-curación    4.2 Unificación   4.3 Balanceo   4.4 Splits│   │
│   │  ┌──────────┐       ┌──────────┐      ┌──────────┐   ┌────────┐│   │
│   │  │-hazelnut │  ──►  │15→6 cats │ ──►  │Under/Over│──►│70/10/20││   │
│   │  │-duplicados│      │Taxonomía │      │sampling  │   │Estratif││   │
│   │  └──────────┘       └──────────┘      └──────────┘   └────────┘│   │
│   │     1,393              1,393           1,022 + aug     1,022    │   │
│   └───────────────────────────────────────────────────────────────┘    │
│                                                      │                  │
│                                                      ▼                  │
│                                              ┌───────────────┐          │
│                                              │  ETAPA 5      │          │
│                                              │  Análisis     │          │
│                                              │  Exhaustivo   │          │
│                                              └───────────────┘          │
│                                                      │                  │
│                                                      ▼                  │
│                                         ┌─────────────────────┐         │
│                                         │   DATASET FINAL     │         │
│                                         │   1,022 imágenes    │         │
│                                         │   6 categorías      │         │
│                                         │   ratio 2.08:1 ✓    │         │
│                                         └─────────────────────┘         │
│                                                                         │
└────────────────────────────────────────────────────────────────────────┘
```

**Tabla sugerida:** *"Evolución del dataset por etapas"*

| Etapa | Operación | Imágenes | Categorías | Ratio Max/Min |
|-------|-----------|----------|------------|---------------|
| Original | - | ~9,142 | 64 | - |
| Etapa 2 | Filtrado | 1,907 | 15 | 24.5:1 |
| Etapa 4.1 | Re-curación | 1,393 | 15 | - |
| Etapa 4.2 | Unificación | 1,393 | 6 | 17.4:1 |
| Etapa 4.3 | Balanceo | 1,022 | 6 | 2.08:1 ✓ |
| **Final** | Splits | **1,022** | **6** | **2.08:1** |

---

### Sección 4: Análisis de Calidad del Dataset

#### Material Gráfico Clave (Alta Prioridad)

Estos gráficos son **imprescindibles** para validar científicamente el dataset:

#### 4.1 Distribución de Categorías

**Figura recomendada:** `category_distribution.png`
- Ubicación: `Final_analysis_curated_dataset_1st_version/outputs/balanced-dataset-analysis-20251114/`
- Muestra la distribución balanceada entre las 6 categorías por split

**Figura alternativa:** `category_proportions.png`
- Pie charts por split mostrando proporciones

#### 4.2 Distribución de Tamaños de Imagen

**Figuras recomendadas:**
- `image_size_distributions.png` - Histogramas de ancho, alto, lado corto y aspect ratio
- `width_vs_height_scatter.png` - Scatter plot que muestra la variabilidad de resoluciones

Ubicación: `Final_analysis_curated_dataset_1st_version/outputs/analysis_imagessizes_plots/`

**Importancia:** Justifica la elección de resolución de entrenamiento (1400×1400 px)

#### 4.3 Análisis de Bounding Boxes

**Figura recomendada:** `bbox_distribution.png`
- Ubicación: `Final_analysis_curated_dataset_1st_version/outputs/balanced-dataset-analysis-20251114/`
- Muestra distribución de width, área y aspect ratio de bboxes

**Figuras complementarias por split:**
- `hist_bbox_area.png` - Distribución de áreas de defectos
- `hist_bbox_aspect_ratio.png` - Variabilidad de formas de defectos

Ubicación: `Final_analysis_curated_dataset_1st_version/outputs/dataset_info/inspect_*/`

**Importancia:** Identifica el problema de bboxes pequeños (<32px) y justifica ajustes de anchors

#### 4.4 Proporción de Augmentación

**Figura recomendada:** `augmentation_distribution.png`
- Ubicación: `Final_analysis_curated_dataset_1st_version/outputs/balanced-dataset-analysis-20251114/`
- Muestra proporción original vs augmentado por split

**Importancia:** Demuestra que la augmentación mantiene proporciones similares entre splits

#### 4.5 Distribución por Dataset de Origen

**Figura recomendada:** `source_dataset_distribution.png`
- Ubicación: `Final_analysis_curated_dataset_1st_version/outputs/balanced-dataset-analysis-20251114/`
- Muestra la contribución de MVTec vs VISION al dataset final

---

### Sección 5: Validación Estadística

#### Contenido Textual
- Test Chi-cuadrado para validar estratificación de splits
- Verificación de no-leakage entre splits
- Validación de distribución uniforme

#### Material Gráfico Recomendado

**Tabla sugerida:** *"Validación estadística de splits"*

| Split | Chi² | p-value | Distribución | Leakage |
|-------|------|---------|--------------|---------|
| Train | 0.006 | 0.999 | ✓ Similar a global | - |
| Val | 0.005 | 0.999 | ✓ Similar a global | 0 imgs |
| Test | 0.013 | 0.999 | ✓ Similar a global | 0 imgs |

---

## Lista de Figuras Prioritarias para la Memoria

### Imprescindibles (Incluir obligatoriamente)

| # | Figura | Fuente | Propósito |
|---|--------|--------|-----------|
| 1 | Diagrama de mapeo de taxonomía | Crear (esquema) | Mostrar unificación de categorías |
| 2 | Diagrama de flujo del pipeline | Crear (esquema) | Visualizar proceso completo |
| 3 | `category_distribution.png` | Etapa 5 | Validar balance de clases |
| 4 | `image_size_distributions.png` | Etapa 5 | Justificar resolución de entrenamiento |
| 5 | `bbox_distribution.png` | Etapa 5 | Identificar características de defectos |
| 6 | `augmentation_distribution.png` | Etapa 5 | Mostrar estrategia de balanceo |

### Recomendadas (Alto valor añadido)

| # | Figura | Fuente | Propósito |
|---|--------|--------|-----------|
| 7 | `width_vs_height_scatter.png` | Etapa 5 | Mostrar variabilidad de resoluciones |
| 8 | `source_dataset_distribution.png` | Etapa 5 | Mostrar composición del dataset |
| 9 | `category_proportions.png` | Etapa 5 | Visualizar proporciones por split |
| 10 | Histogramas de bbox por split | Etapa 5 | Detallar distribución de defectos |

### Opcionales (Para mayor detalle)

| # | Figura | Fuente | Propósito |
|---|--------|--------|-----------|
| 11 | Mosaico de ejemplos de defectos | Datasets originales | Ilustrar tipos de defectos |
| 12 | Ejemplos de augmentación | Dataset final | Mostrar transformaciones aplicadas |
| 13 | Distribución de resoluciones original | Etapa 3 | Comparar antes/después de normalización |

---

## Tablas de Métricas Esenciales

### Tabla 1: Resumen del Dataset Final

```markdown
| Característica | Valor |
|----------------|-------|
| Total imágenes | 1,022 |
| Total anotaciones | 1,354 |
| Categorías | 6 |
| Ratio máx/min | 2.08:1 |
| Resolución media | 1,695 × 1,406 px |
| Área media de imagen | 3.34 MP |
```

### Tabla 2: Distribución por Split

```markdown
| Split | Imágenes | Anotaciones | % | Augmentadas |
|-------|----------|-------------|---|-------------|
| Train | 715 | 944 | 70% | 35.9% |
| Val | 102 | 145 | 10% | 38.2% |
| Test | 205 | 265 | 20% | 35.1% |
```

### Tabla 3: Distribución por Categoría

```markdown
| Categoría | Total | Train | Val | Test | % |
|-----------|-------|-------|-----|------|---|
| PERFORACIONES | 328 | 242 | 26 | 60 | 24.2% |
| NORMAL | 300 | 210 | 30 | 60 | 22.2% |
| ROTURA_FRACTURA | 211 | 138 | 33 | 40 | 15.6% |
| DEFORMACIONES | 195 | 136 | 21 | 38 | 14.4% |
| RAYONES_ARANAZOS | 162 | 111 | 17 | 34 | 12.0% |
| CONTAMINACION | 158 | 107 | 18 | 33 | 11.7% |
```

### Tabla 4: Estadísticas de Bounding Boxes

```markdown
| Métrica | Train | Val | Test |
|---------|-------|-----|------|
| N BBoxes | 944 | 145 | 265 |
| Width (mediana) | 222.4 px | 224.0 px | 314.0 px |
| Height (mediana) | 132.5 px | 112.0 px | 175.0 px |
| Área (mediana) | 6,082 px² | 7,435 px² | 7,914 px² |
| Aspect Ratio | 1.00 | 1.00 | 1.00 |
| % Width < 32px | 23.4% | 16.6% | 17.0% |
```

---

## Recomendaciones Adicionales para la Memoria

### Sección de Metodología

1. **Incluir diagrama de flujo completo** del pipeline (Figura 4 sugerida)
2. **Justificar cada decisión** con métricas cuantitativas
3. **Documentar criterios de exclusión** (ej. hazelnut, duplicados)

### Sección de Resultados

1. **Presentar antes/después** del balanceo (ratio 24.5:1 → 2.08:1)
2. **Validar estadísticamente** la estratificación de splits
3. **Identificar limitaciones** (bboxes pequeños, necesidad de ajustar anchors)

### Sección de Discusión

1. **Comparar con otros datasets** de detección de defectos
2. **Justificar la taxonomía** frente a alternativas
3. **Discutir implicaciones** para el entrenamiento de modelos

### Anexos Técnicos

1. **Esquema JSON** del formato COCO híbrido utilizado
2. **Configuración de augmentación** aplicada
3. **Scripts de reproducibilidad** (referencia a `flujo_curacion_dataset/`)

---

## Conclusión

El proceso de curación del dataset representa un trabajo riguroso y científicamente fundamentado que merece documentación detallada en la memoria técnica. Los elementos visuales y tablas propuestas permiten:

1. **Demostrar la calidad** del trabajo realizado
2. **Facilitar la reproducibilidad** del proceso
3. **Proporcionar evidencia cuantitativa** de las decisiones tomadas
4. **Comunicar eficazmente** el esfuerzo de preparación de datos

La inclusión de estos elementos elevará significativamente la calidad de la memoria técnica y demostrará un enfoque profesional en el tratamiento de datos para Machine Learning.

---

**Documento generado:** Diciembre 2025  
**Propósito:** Recomendaciones para documentación en memoria técnica TFG

