# TFG: Vision Transformers para Detección de Anomalías Industriales

## 1. PROPUESTA DE INVESTIGACIÓN

### Título
**"Vision Transformers Adaptados para Detección de Anomalías en Componentes Industriales: Transfer Learning y Validación Práctica"**

### Objetivo Principal
Desarrollar un sistema de detección de anomalías industriales basado en Vision Transformers (ViT) mediante transfer learning, con validación práctica en componentes electrónicos reales y comparación robusta con métodos tradicionales.

### Justificación Técnica
- **ViTs en Industria**: Aplicación emergente de transformers para inspección visual industrial
- **Transfer Learning**: Aprovechamiento de modelos pre-entrenados para adaptación eficiente a dominios específicos
- **Gap Identificado**: Necesidad de validación práctica de ViTs en componentes industriales reales
- **Benchmarking**: Comparación sistemática con métodos tradicionales de detección

## 2. ESTADO DEL ARTE

### 2.1 Vision Transformers en Detección de Anomalías
- Aplicación reciente y creciente en computer vision industrial
- Capacidad superior de capturar relaciones globales en imágenes
- Interpretabilidad mejorada mediante attention mechanisms

### 2.2 Oportunidad de Investigación
- **Timing Apropiado**: ViTs maduros y estables para aplicaciones prácticas
- **Datasets Disponibles**: Múltiples datasets industriales etiquetados recientes
- **Hardware Accesible**: RTX 4070 suficiente para ViT-Base inference y fine-tuning

## 3. DATASETS Y RECURSOS

### 3.1 Dataset Principal: VISION
- **Tamaño**: 4,000 imágenes etiquetadas, 44 tipos de defectos
- **Ventajas**: Dataset industrial específico con anotaciones detalladas
- **Acceso**: Disponible en Hugging Face

### 3.2 Datasets Complementarios
- **MVTec AD**: 5,354 imágenes, benchmark estándar detección anomalías
- **3CAD**: 27,039 imágenes alta resolución, componentes 3C
- **Componentes Escuela**: 50-100 imágenes capturadas para demo práctica

## 4. METODOLOGÍA TÉCNICA

### 4.1 Arquitectura Propuesta

```python
# Pipeline técnico
1. ViT Pre-entrenado: Modelo base ImageNet adaptado
2. Fine-tuning: Entrenamiento en datasets industriales etiquetados
3. Transfer Learning: Adaptación a componentes específicos escuela
4. Evaluación: Benchmarking vs métodos tradicionales
5. Demo: Sistema tiempo real con interpretabilidad visual
```

### 4.2 Componentes Técnicos
- **Modelo Base**: ViT-Base/16 (Hugging Face Transformers)
- **Framework**: PyTorch + Timm para implementaciones optimizadas
- **Baselines**: ResNet-50, EfficientNet, métodos clásicos
- **Interpretabilidad**: Attention rollout, GradCAM

## 5. CRONOGRAMA (12 SEMANAS)

### Fase 1: Setup y Transfer Learning (4 semanas)
- **Semana 1**: Setup entorno, descarga datasets VISION + MVTec AD
- **Semana 2**: Implementación ViT baseline y pipeline fine-tuning
- **Semana 3**: Fine-tuning en dataset VISION industrial
- **Semana 4**: Validación en MVTec AD, optimización hiperparámetros

### Fase 2: Benchmarking y Baselines (4 semanas)
- **Semana 5**: Implementación baselines (ResNet, EfficientNet)
- **Semana 6**: Entrenamiento comparativo en mismo dataset
- **Semana 7**: Evaluación sistemática y métricas comparativas
- **Semana 8**: Análisis interpretabilidad (attention maps)

### Fase 3: Demo Práctica y Validación (4 semanas)
- **Semana 9**: Captura y preparación componentes electrónicos escuela
- **Semana 10**: Transfer learning a componentes reales
- **Semana 11**: Desarrollo demo visual tiempo real
- **Semana 12**: Documentación final y preparación defensa

## 6. CASOS DE USO PRÁCTICOS

### 6.1 Componentes Objetivo para Demo
- **PCBs**: Placas circuito impreso (soldaduras defectuosas, pistas dañadas)
- **Conectores**: Puertos USB, conectores (desgaste, corrosión)  
- **Carcasas**: Dispositivos electrónicos (grietas, rayones)
- **Componentes SMD**: Resistencias, capacitores (posicionamiento incorrecto)

### 6.2 Demo Visual en Tiempo Real
```python
# Sistema demo interactivo
1. Captura: Webcam/cámara escuela → imagen componente
2. Procesamiento: ViT modelo entrenado → predicción anomalía
3. Visualización: Attention maps → regiones anómalas destacadas
4. Interface: Dashboard confianza + tipo anomalía detectada
5. Comparación: Resultados ViT vs baselines side-by-side
```

### 6.3 Métricas Demo
- **Precisión**: Accuracy, F1-score, AUC-ROC detección
- **Velocidad**: FPS procesamiento tiempo real RTX 4070
- **Interpretabilidad**: Calidad attention maps para localización defectos
- **Robustez**: Rendimiento diferentes condiciones iluminación/ángulos

## 7. RECURSOS Y VIABILIDAD

### 7.1 Hardware Disponible
- **GPU**: RTX 4070 (12GB VRAM, suficiente ViT-Base fine-tuning)
- **RAM**: 16GB mínimo para datasets y modelos
- **Storage**: 100GB datasets + modelos entrenados

### 7.2 Software (Open Source)
```python
# Stack tecnológico completo
- PyTorch: Framework principal ML
- Transformers: Hugging Face ViT implementations
- Timm: Modelos vision optimizados
- OpenCV: Procesamiento imagen demo
- Streamlit: Interface demo interactiva
```

### 7.3 Datasets (Públicos)
- VISION: Hugging Face acceso directo
- MVTec AD: Dataset benchmark público estándar
- 3CAD: GitHub repository disponible

## 8. CONTRIBUCIONES ESPERADAS

### 8.1 Técnicas
- **Aplicación práctica** ViT transfer learning detección anomalías industriales
- **Benchmarking sistemático** ViT vs métodos tradicionales componentes reales
- **Análisis interpretabilidad** attention mechanisms para localización defectos
- **Pipeline reproducible** fine-tuning ViT datasets industriales

### 8.2 Prácticas
- **Sistema funcional** detección tiempo real componentes escuela
- **Demo interactiva** con visualización attention maps
- **Código open source** implementación completa reproducible
- **Validación experimental** rendimiento múltiples tipos componentes

## 9. GESTIÓN DE RIESGOS

### 9.1 Riesgos Técnicos y Mitigaciones
- **Convergencia fine-tuning**: ViT pre-entrenado + datasets etiquetados garantizan baseline sólido
- **Calidad componentes escuela**: Validación previa en datasets estándar (MVTec AD)
- **Hardware limitaciones**: ViT-Base optimizado específicamente para RTX 4070

### 9.2 Riesgos Temporales
- **Cronograma ajustado**: Enfoque implementación vs investigación teórica extensiva
- **Scope realista**: Transfer learning vs entrenamiento desde cero
- **Fallback plan**: MVTec AD como validación si componentes escuela presentan dificultades

## 10. RESULTADOS ESPERADOS

### 10.1 Métricas Técnicas Objetivo
- **AUC-ROC**: >0.90 en MVTec AD (competitivo estado del arte)
- **Velocidad**: >15 FPS inferencia RTX 4070
- **Tamaño modelo**: <500MB deployment eficiente

### 10.2 Entregables
- **Modelo entrenado**: ViT adaptado detección anomalías industriales
- **Demo interactiva**: Detección tiempo real + attention visualization
- **Benchmarking completo**: Comparativa cuantitativa vs baselines
- **Código reproducible**: Pipeline completo GitHub

### 10.3 Demostración Final
- **Interface web**: Upload imagen → detección instantánea anomalías
- **Visualización interpretable**: Attention maps sobre regiones defectuosas
- **Dashboard métricas**: Confianza, tipo anomalía, tiempo procesamiento
- **Casos reales**: 15-20 componentes diferentes escuela validados

## 11. CONCLUSIONES

Este TFG aplica Vision Transformers a detección de anomalías industriales mediante un enfoque pragmático de transfer learning. El scope está dimensionado para 12 semanas de desarrollo intensivo con resultados garantizados y demo práctica funcional.

**Fortalezas del enfoque**:
- Tecnología probada (ViT) con aplicación práctica novedosa
- Datasets etiquetados disponibles minimizan riesgo etiquetado manual
- Hardware suficiente para implementación completa
- Demo tangible con componentes reales

**Viabilidad técnica**:
- Transfer learning reduce complejidad vs entrenamiento desde cero
- Múltiples datasets backup garantizan datos suficientes
- Baselines claros para comparación objetiva
- Cronograma realista enfocado en resultados

**Impacto esperado**:
- Contribución práctica aplicación ViTs industria
- Sistema funcional demostrable
- Base sólida para extensiones futuras (SSL, multi-escala)
- Benchmarking valioso para comunidad investigación

---

**Resumen recursos**:
- **Hardware**: RTX 4070 + 16GB RAM (disponible)
- **Tiempo**: 12 semanas desarrollo (realista)
- **Presupuesto**: $0 (todo open source/público)
- **Datos**: Sin etiquetado manual masivo requerido