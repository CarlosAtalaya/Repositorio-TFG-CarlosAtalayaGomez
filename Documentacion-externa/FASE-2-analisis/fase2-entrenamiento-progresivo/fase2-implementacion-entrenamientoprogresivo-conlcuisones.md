# 4. Experimentación Fase 2: Refinamiento Multimodal Progresivo

## 4.1. Justificación Teórica y Planteamiento de la Hipótesis Multimodal

### 4.1.1. Evaluación Crítica del Baseline Visual (Fase 1)
La primera fase de esta investigación culminó con la implementación y entrenamiento de la arquitectura DEIMv2 respaldada por el backbone DINOv3, alcanzando una métrica de referencia (*baseline*) de **0.785 mAP (IoU 0.50:0.95)**. Si bien este resultado valida empíricamente la superioridad de los Vision Transformers sobre las arquitecturas convolucionales clásicas (ResNet, EfficientNet) para la extracción de características en dominios industriales, un análisis granular de los errores de clasificación revela limitaciones intrínsecas al paradigma puramente visual.

El examen de las matrices de confusión y los casos de falsos negativos pone de manifiesto una dificultad recurrente en la discriminación de defectos que, aunque semánticamente distintos, comparten una topología visual extremadamente similar. Específicamente, se observa una alta tasa de confusión entre las clases **"Rotura/Fractura"** y **"Rayones Profundos"**. Desde la perspectiva del espacio de características visuales extraído por el encoder, ambas anomalías se manifiestan como discontinuidades lineales de alta frecuencia y bajo contraste sobre la superficie metálica. La red neuronal, al carecer de un contexto superior, tiende a agrupar estas instancias basándose exclusivamente en sus propiedades geométricas (bordes, gradientes de píxeles), ignorando las sutilezas estructurales que diferencian una ruptura del material de una abrasión superficial.

Este fenómeno sugiere que el modelo ha alcanzado una **asíntota de rendimiento visual**: aumentar la capacidad del modelo o el tiempo de entrenamiento bajo el mismo paradigma supervisado probablemente resultaría en *overfitting* (sobreajuste) a las texturas del conjunto de entrenamiento, sin resolver la ambigüedad fundamental entre clases morfológicamente análogas.

### 4.1.2. La Hipótesis de la Información Ortogonal
Para superar esta barrera, esta investigación postula la **Hipótesis de la Ortogonalidad Semántica**. Esta hipótesis sostiene que la incorporación de descripciones en lenguaje natural no actúa meramente como un dato redundante, sino que introduce un vector de información ortogonal (independiente) al visual.

Mientras que el encoder visual (DINOv3) procesa la imagen en términos de *"geometría, textura y patrones espaciales"*, un encoder textual procesa la definición del defecto en términos de *"atributos, causalidad y naturaleza del objeto"*. Por ejemplo, la descripción *"Fractured material with jagged edges and separation"* (Material fracturado con bordes dentados y separación) contiene claves semánticas que fuerzan al modelo a buscar características específicas (separación física, irregularidad de borde) que podrían pasar desapercibidas en una búsqueda puramente de texturas.

### 4.1.3. Propuesta de Refinamiento Progresivo (Residual Learning)
En consecuencia, la Fase 2 de este proyecto no busca reemplazar el detector visual, que ya ha demostrado ser altamente competente, sino **refinarlo**. Se propone una arquitectura de fusión multimodal que opera bajo el principio de **Aprendizaje Residual**:

$$F(x)_{final} = F(x)_{visual} + \Delta(x)_{lenguaje}$$

Donde el componente de lenguaje $\Delta(x)$ actúa como un mecanismo de atención semántica guiada. El objetivo científico es demostrar que, al proyectar los embeddings visuales y textuales en un espacio latente común, es posible "empujar" los clusters de clases confusas (ej. grietas vs. rayones) para separarlos, aumentando el margen de decisión del clasificador en los casos límite (*edge cases*) donde la evidencia visual es ambigua. Esta estrategia busca mejorar específicamente el *Recall* en defectos críticos sin degradar la precisión global ya consolidada en la Fase 1.

---

## 4.2. Arquitectura del Sistema Multimodal: Diseño del Espacio Latente Compartido

Para operacionalizar la Hipótesis de la Ortogonalidad Semántica, se diseñó una extensión modular sobre la arquitectura base DEIMv2. El sistema resultante, denominado **DEIMv2-Multimodal**, integra tres componentes funcionales que operan en cascada: un backbone visual preentrenado, un encoder textual congelado y un mecanismo de fusión aprendible.

### 4.2.1. Extracción de Características y Definición de Embeddings
El procesamiento de la información se bifurca en dos flujos paralelos que convergen en la etapa de clasificación final:

1.  **Flujo Visual (Representación Geométrica):**
    Se utiliza el backbone **DINOv3** integrado en DEIMv2 para extraer representaciones densas de las regiones de interés (RoIs). Sea $x_{img}$ una imagen de entrada, el encoder visual $E_v$ genera un conjunto de descriptores visuales $V \in \mathbb{R}^{N \times D_v}$, donde $N$ es el número de *queries* o propuestas de objetos y $D_v=256$ es la dimensión del espacio latente visual. Estos vectores encapsulan información de bajo nivel (textura, bordes) y alto nivel (forma, estructura).

2.  **Flujo Textual (Representación Semántica):**
    Para inyectar conocimiento externo, se emplea un **Text Encoder** basado en la arquitectura CLIP (Contrastive Language-Image Pre-training). Se define un conjunto de $K$ prompts enriquecidos, $P = \{p_1, p_2, ..., p_K\}$, donde cada $p_i$ describe los atributos visuales distintivos de la clase $i$ (e.g., *"Fractured material, jagged edges..."*).
    Estos prompts son procesados por el encoder textual $E_t$ para generar una matriz de embeddings textuales $T \in \mathbb{R}^{K \times D_t}$, donde $K=6$ (clases de defectos) y $D_t=512$ (dimensión del espacio CLIP). Es crucial destacar que los pesos de $E_t$ se mantienen **congelados** (*frozen*) durante el entrenamiento para preservar la riqueza semántica aprendida en su preentrenamiento a gran escala, evitando el *catastrophic forgetting* dado el tamaño reducido del dataset industrial.

### 4.2.2. Módulo de Proyección y Espacio Latente Común
Dado que $D_v \neq D_t$ (256 vs 512), existe una discrepancia dimensional que impide la comparación directa. Para resolver esto, se implementa un **Módulo de Fusión (`MultimodalFusionModule`)** que proyecta los embeddings visuales al espacio semántico del texto.

Se define una transformación lineal aprendible $W_{proj} \in \mathbb{R}^{D_v \times D_t}$:
$$V'_{proj} = V \cdot W_{proj}$$

Esta operación alinea geométricamente las características visuales con las textuales. Posteriormente, para garantizar la estabilidad numérica y la validez de la métrica de similitud, ambos conjuntos de vectores se normalizan en la hiperesfera unitaria ($L_2$ normalization):
$$\hat{v}_i = \frac{v'_{proj, i}}{\|v'_{proj, i}\|_2}, \quad \hat{t}_j = \frac{t_j}{\|t_j\|_2}$$

### 4.2.3. Mecanismo de Atención por Similitud Coseno
La alineación semántica se calcula mediante el producto punto entre los vectores normalizados, lo que equivale a la **Similitud Coseno**. Se genera una matriz de afinidad $S \in \mathbb{R}^{N \times K}$ que representa cuánto se "parece" visualmente cada región propuesta a la descripción textual de cada defecto:

$$S_{i,j} = \frac{\hat{v}_i \cdot \hat{t}_j^\top}{\tau}$$

Donde $\tau$ es un parámetro de temperatura aprendible (inicializado en 0.07) que regula la entropía de la distribución de similitud, controlando la nitidez de las asignaciones clase-objeto.

### 4.2.4. Inyección de Corrección Residual (Logit Refinement)
La innovación central de esta arquitectura reside en cómo se utiliza esta matriz $S$. En lugar de sustituir al clasificador visual original, la información semántica se inyecta como un **término de corrección residual** sobre los *logits* de clasificación base de DEIMv2 ($L_{base}$).

La predicción final $L_{final}$ se modela como:
$$L_{final} = L_{base} + \alpha \cdot S$$

Donde $\alpha$ es un escalar aprendible (*gating parameter*). Esta formulación matemática tiene implicaciones profundas:
1.  **Preservación del Conocimiento:** Si $\alpha \to 0$, el modelo colapsa a su versión unimodal original, garantizando que el rendimiento nunca sea teóricamente inferior al *baseline*.
2.  **Desempate Semántico:** En situaciones de ambigüedad visual donde $L_{base}$ presenta valores similares para dos clases (ej. grieta vs. rayón), el término $\alpha \cdot S$ actúa como un "voto de calidad", amplificando la clase cuya descripción textual resuena más con la morfología detectada.

---

## 4.3. Desafíos de Convergencia: El Fenómeno de Interferencia y la Estrategia "Zero-Start"

La integración de modalidades heterogéneas (visión y lenguaje) sobre un backbone preentrenado presenta desafíos significativos en la optimización. Durante las fases experimentales preliminares, se identificó un fenómeno crítico de inestabilidad que denominamos **Interferencia Destructiva en la Inicialización**, el cual comprometía la integridad de las representaciones visuales aprendidas en la Fase 1.

### 4.3.1. Diagnóstico del Colapso de Rendimiento (Destructive Interference)
En la implementación ingenua inicial, el módulo de fusión se inicializó siguiendo el estándar de Xavier/Kaiming para los pesos $W_{proj}$ y estableciendo un valor arbitrario para el factor de escala (e.g., $\alpha_{init} = 0.1$). Bajo estas condiciones, al inicio del entrenamiento ($t=0$), el término de corrección multimodal $\alpha \cdot S$ introdujo una perturbación estocástica de magnitud considerable sobre los *logits* calibrados del detector base.

Sea $L_{base}$ la salida del detector óptimo de la Fase 1. La salida inicial del sistema multimodal fue:
$$L_{final}^{(t=0)} = L_{base} + \text{Ruido}(\mu \approx 0, \sigma_{proj})$$

Empíricamente, esto resultó en una degradación inmediata del rendimiento, con el mAP descendiendo de **0.785** a **~0.670** en la primera época, y una oscilación severa en la función de pérdida. El modelo se vio forzado a dedicar las primeras épocas no a aprender nuevas correlaciones semánticas, sino a "limpiar" el ruido inyectado, un proceso ineficiente que aumenta el riesgo de caer en mínimos locales subóptimos.

### 4.3.2. Implementación de la Estrategia "Zero-Start"
Para mitigar este riesgo y garantizar un **Aprendizaje Monótono No Decreciente**, se desarrolló e implementó la estrategia de inicialización **"Zero-Start"**. Esta técnica se fundamenta en el principio de identidad al inicio del *fine-tuning*.

La implementación técnica impone dos restricciones fuertes en el estado inicial de los parámetros:

1.  **Gating Parameter Nulo:** El parámetro escalar $\alpha$ se inicializa explícitamente en $0.0$.
    $$\alpha^{(t=0)} := 0$$
    Esto anula algebraicamente la contribución del ramal textual en la iteración inicial, garantizando que $L_{final}^{(t=0)} \equiv L_{base}$. En consecuencia, el modelo comienza el entrenamiento preservando intacto el mAP de 0.785 del *baseline*.

2.  **Inicialización de Baja Varianza:** Los pesos de la proyección lineal $W_{proj}$ se inicializan desde una distribución normal con desviación estándar mínima ($\sigma=0.01$) y sesgos en cero. Esto asegura que, conforme $\alpha$ comienza a crecer por acción del gradiente, los vectores proyectados $V'_{proj}$ tengan magnitudes controladas, evitando "explosiones" en la métrica de similitud coseno.

### 4.3.3. Dinámica del Gradiente Resultante
Bajo la estrategia "Zero-Start", la actualización del parámetro $\alpha$ está gobernada exclusivamente por la señal del gradiente de la función de pérdida ($\mathcal{L}$):
$$\frac{\partial \mathcal{L}}{\partial \alpha} = \frac{\partial \mathcal{L}}{\partial L_{final}} \cdot S$$

El optimizador (AdamW) solo incrementará el valor de $\alpha$ (dándole peso al texto) si y solo si la matriz de similitud $S$ contiene información correlacionada negativamente con el error de clasificación actual. Esto transforma el proceso de *fine-tuning* en una búsqueda selectiva: el modelo aprende a "escuchar" al texto progresivamente y solo para aquellas instancias donde la descripción semántica ayuda a reducir la entropía de la predicción, actuando efectivamente como un mecanismo de **Atención Residual Adaptativa**.

---

## 4.4. Estrategia de Entrenamiento Progresivo (Curriculum Learning)

La optimización de arquitecturas profundas multimodales requiere una gestión cuidadosa del flujo de gradientes para equilibrar la plasticidad (capacidad de aprender nueva información semántica) y la estabilidad (capacidad de retener el conocimiento visual previo). Para lograr este equilibrio, se diseñó un protocolo de entrenamiento por etapas fundamentado en el paradigma de **Curriculum Learning** (Aprendizaje Curricular).

### 4.4.1. Preservación del Conocimiento Visual (Backbone Freezing)
La primera decisión estratégica consistió en congelar (*freeze*) la gran mayoría de los parámetros del modelo durante la Fase 2. Específicamente, se bloquearon los gradientes ($\nabla_\theta = 0$) para:
1.  El backbone visual **DINOv3**: Sus pesos contienen filtros de extracción de características altamente robustos y generalistas, entrenados en la Fase 1. Modificarlos con un dataset pequeño de pares imagen-texto induciría un deterioro rápido de la calidad de los mapas de características.
2.  El encoder de **DEIMv2** (Transformer Encoder): Responsable de la relación espacial entre objetos, conocimiento que debe permanecer invariante.

Matemáticamente, definimos el conjunto de parámetros entrenables $\Theta_{train}$ como un subconjunto estricto de los parámetros totales $\Theta_{total}$:
$$\Theta_{train} = \{ \theta_{fusion}, \theta_{score\_head}, \theta_{class\_embed} \} \subset \Theta_{total}$$

Esto redujo el espacio de búsqueda de optimización de millones de parámetros a aproximadamente **140,000 parámetros críticos**. Esta reducción drástica no solo disminuye el coste computacional, sino que actúa como un regularizador estructural fuerte, impidiendo que el modelo "memorice" el ruido del dataset de entrenamiento.

### 4.4.2. Foco en la Alineación Semántica
Al restringir la actualización de pesos exclusivamente al **Módulo de Fusión** y a las **Cabeceras de Clasificación**, se fuerza al optimizador a resolver una tarea muy específica: encontrar la transformación lineal óptima que mapee la geometría visual existente a la semántica textual propuesta.

Este enfoque desacopla el problema de "aprender a ver" del problema de "aprender a nombrar". En la Fase 1, el modelo aprendió a discriminar anomalías basándose en diferencias de píxeles. En esta Fase 2, el modelo aprende a re-ponderar esas anomalías basándose en su congruencia con las descripciones textuales. Es análogo a enseñar a un experto visual (que ya sabe detectar grietas) a utilizar un nuevo vocabulario técnico para clasificarlas con mayor precisión, sin necesidad de volver a enseñarle qué es una grieta.

### 4.4.3. Prevención del Olvido Catastrófico
Un riesgo inherente al *fine-tuning* secuencial es el **Olvido Catastrófico**. Para mitigar esto, se adoptaron hiperparámetros conservadores en el protocolo de optimización:
* **Tasa de Aprendizaje Reducida:** Se empleó un *learning rate* de $\eta = 5 \times 10^{-5}$, un orden de magnitud inferior al utilizado en el entrenamiento base.
* **Ciclo de Entrenamiento Corto:** Se limitó el horizonte de entrenamiento a un máximo de 100 épocas con una política de *Early Stopping* agresiva (Paciencia = 20). La hipótesis subyacente es que la alineación semántica es una tarea de convergencia rápida; si el modelo no logra alinear visión y texto en pocas épocas, prolongar el entrenamiento solo aumentaría el riesgo de *overfitting*.

---

## 4.5. Protocolo de Evaluación Híbrida: Más allá del mAP

La adopción de una métrica de evaluación estándar como el **mAP** resulta insuficiente para capturar la verdadera eficacia operativa de un sistema de inspección industrial, donde el coste de los errores es asimétrico (un Falso Negativo tiene repercusiones más graves que un Falso Positivo).

### 4.5.1. Insuficiencia del mAP Global
Se observó que el mAP global está dominado estadísticamente por las clases mayoritarias y "fáciles" (e.g., *Normal* o *Perforaciones*). Mejoras significativas en las clases minoritarias y difíciles (e.g., incrementar el *Recall* de *Rotura*) a menudo tienen un impacto despreciable en el promedio global. Guiarse exclusivamente por el mAP global para el *Early Stopping* conlleva el riesgo de descartar modelos que son **operativamente superiores** solo porque su precisión estadística promedio no aumenta.

### 4.5.2. Formulación del Índice de Desempeño Compuesto ($S_{composite}$)
Para alinear el proceso de optimización con los objetivos industriales, se diseñó e implementó una métrica de evaluación compuesta para el criterio de parada y selección de checkpoints. Se define el **Score Híbrido** ($S_{composite}$) como una combinación lineal convexa:

$$S_{composite} = \lambda_1 \cdot mAP_{coco} + \lambda_2 \cdot AR_{critical}$$

Donde:
* $mAP_{coco}$: Precisión Media Global bajo métricas estándar.
* $AR_{critical}$: *Average Recall* (Promedio de Sensibilidad) calculado exclusivamente sobre el subconjunto de clases de alto riesgo ($C_{crit} \subset C_{total}$).
* $\lambda_1=0.7, \lambda_2=0.3$: Coeficientes de ponderación que priorizan la estabilidad general del modelo (70%) pero otorgan un peso decisivo (30%) a la capacidad de recuperación de defectos.

### 4.5.3. Definición de Clases Críticas y Maximización del Recall
El subconjunto $C_{crit}$ se definió seleccionando aquellas tipologías de defectos que presentan mayor ambigüedad visual o mayor riesgo estructural:
1.  **Rotura/Fractura (ID 2):** Debido a su similitud con rayones superficiales y su criticidad mecánica.
2.  **Rayones/Arañazos (ID 3):** Por su alta varianza intra-clase.
3.  **Contaminación (ID 5):** Por su baja saliencia visual frente al fondo metálico.

Este protocolo garantiza que la Fase 2 no se considere exitosa simplemente por mejoras numéricas globales, sino por demostrar una capacidad superior en la tarea de **desambiguación semántica**, rescatando defectos verdaderos que el modelo visual puro descartaba por incertidumbre.