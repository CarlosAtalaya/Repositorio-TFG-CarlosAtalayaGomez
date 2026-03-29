# Propuesta de estructura para la defensa oral del TFG (presentación + demo interactiva)

**Audiencia objetivo:** tribunal de la Escuela de Ingeniería con formación técnica general, **sin necesidad de ser experto en visión por computador o aprendizaje profundo**. El mensaje debe ser: **qué problema industrial hay**, **qué se ha hecho**, **por qué importa**, **qué resultado objetivo se obtiene**, y **cómo se comprueba**.

**Recursos disponibles:**

- Memoria cerrada en `TFG_Carlos_Atalaya_25-26/`.
- **Herramienta Streamlit** (`Documentacion-externa/FASE-1-analisis/herramienta_comparativa/`, `dashboard.py`): panel con vistas **Inicio**, **Línea temporal**, **Explorador**, **Comparativa**, **Visualizaciones**, **Conclusiones**.

**Principio de diseño:** la presentación (PowerPoint, Google Slides, LaTeX Beamer, etc.) lleva la **historia y las ideas**; Streamlit lleva la **evidencia interactiva** (números, curvas, comparación de cajas y predicciones). Evita saturar diapositivas con tablas enormes: **una cifra o una idea por diapositiva**; el detalle vive en la demo.

**Duración orientativa (ajustar al cronograma oficial):** 15–20 minutos de exposición + 5–10 minutos de demo + tiempo de preguntas.

---

## 1. Guion general (storyline)

1. **Gancho (30 s):** inspección industrial = calidad y coste; el cuello de botella no es solo “una cámara”, sino **datos de defectos escasos** y sistemas que **no generalizan**.
2. **Problema en una frase:** detectar y clasificar defectos en imágenes con **pocas anotaciones** y defectos **muy pequeños o globales** a la vez.
3. **Idea clave sin jerga:** las redes clásicas (CNN) miran bien el “vecindario” de cada punto; los **Transformers de visión** pueden relacionar **regiones lejanas** (útil cuando el defecto es un patrón respecto a toda la pieza).
4. **Qué has hecho:** dataset unificado público + entrenamiento de **dos CNN** y un detector **DEIMv2** con **preentrenamiento auto-supervisado (DINOv3)**.
5. **Resultado principal:** mejora fuerte en **mAP** frente a CNN; **validación** de que no es solo “subir resolución”.
6. **Cómo lo demuestras:** panel web con **comparativas** y **imágenes reales** predichas por cada modelo.
7. **Cierre:** limitaciones (tamaño del dataset, dominio), trabajo futuro breve, agradecimiento.

---

## 2. Mapa diapositiva ↔ demo Streamlit

| Bloque PPT | Contenido en la diapositiva (mínimo) | Qué mostrar en Streamlit (cuando toque) |
|------------|--------------------------------------|----------------------------------------|
| Título | Título del TFG, nombre, tutor/es, escuela, curso | — |
| Motivación | Foto o esquema de línea de inspección; 2 viñetas: escasez de defectos + necesidad de automatizar | — |
| Objetivo | Una frase de objetivo general + 2–3 viñetas de objetivos específicos (comparar arquitecturas, dataset curado, herramienta) | Opcional: **Inicio** (resumen ejecutivo ya está en la app) |
| Contexto ML (1 diapositiva) | Esquema: imagen → modelo → cajas/clases; “CNN = ventana local” vs “ViT = atención global” (**sin ecuaciones**) | — |
| Datos | De dónde vienen (VISION + MVTec), **6 categorías**, tamaño aproximado (**~1000 imágenes**), que los splits están bien hechos | — |
| Metodología experimental | Diagrama de **4 fases** (dataset → CNN → ViT → validación CNN a alta resolución); mencionar métrica **mAP** como “puntuación global de acierto en detección” | **Línea temporal**: narrar el recorrido por fases con las tarjetas |
| Resultados globales | **Una** figura tipo barras: mAP de la mejor CNN vs DEIMv2 (números ya en memoria) | **Comparativa**: gráfico mAP de todos los experimentos; filtrar por fase si hace falta |
| Resultado “arquitectura vs resolución” | Mensaje: CNN casi no mejora o empeora a 1024×1024; el ViT sí aprovecha resolución y arquitectura | **Explorador**: elegir experimentos CNN 1024 vs ViT 1024 y leer mAP |
| Robustez (opcional, 1 diapositiva) | Umbrales de confianza más altos aún dejan mAP alto (mensaje de confianza operativa) | **Explorador** (DEIMv2): selector de umbral **0,15 / 0,75 / 0,90** |
| Demo “por qué importa” | Titular: “Más allá del número: qué ve cada modelo” | **Visualizaciones**: misma imagen, cajas de referencia vs CNN vs ViT; **slider** de umbral |
| Herramienta | Una captura del panel + una frase: reproducibilidad y apoyo a decisión | Navegación rápida por **Inicio → Conclusiones** |
| Conclusiones | 3 viñetas: hipótesis confirmada, limitaciones (dataset, dominio), futuro (más datos / VLMs) | — |
| Preguntas | Diapositiva final con título, email, enlace al repo si procede | — |

---

## 3. Estructura detallada de la presentación (orden sugerido)

### Apertura (2–3 diapositivas)

1. **Título** (título oficial del TFG, autor, tutores, centro).
2. **Problema industrial** (1 figura o icono): control de calidad, coste del fallo, inspección manual limitada.
3. **Pregunta de investigación** en una línea: *¿Puede un detector basado en Vision Transformer superar a las CNN clásicas en detección de defectos con datos limitados, y bajo qué condiciones?*

### Núcleo técnico “ligero” (4–6 diapositivas)

4. **Qué es un defecto aquí:** detección de objetos con cajas (bounding boxes) y clases; referencia visual simple.
5. **Por qué no basta una CNN “de libro”:** campo receptivo local; ejemplo intuitivo (defecto que solo se entiende viendo la pieza entera).
6. **Enfoque propuesto:** DEIMv2 + backbone preentrenado con SSL (DINOv3) — **una frase cada concepto**, logotipo o diagrama del memoria si ya lo tienes.
7. **Dataset:** fuentes, 6 clases con **nombres en castellano** en la diapositiva (y equivalencia en inglés si el tribunal lee papers en inglés).
8. **Diseño experimental:** Fase 1 CNN → Fase 2 ViT → Fase 3 misma resolución para CNN; **sin listar hiperparámetros** salvo que pregunten.

### Resultados (2–4 diapositivas)

9. **Tabla o gráfico único** con mAP de las configuraciones estrella (mejor CNN vs mejor DEIMv2).
10. **Mensaje de la Fase 3:** la brecha **no** se explica solo por más píxeles.
11. (Opcional) **Robustez** con umbrales o precisión/exhaustividad a alto nivel.

### Demo interactiva (5–10 minutos)

12. **Diapositiva puente:** “Demostración en vivo: panel de comparación y predicciones.”
    - Arrancar Streamlit antes de la defensa (`streamlit run dashboard.py` desde la carpeta de la herramienta).
    - Pantalla dual recomendable: **PPT a pantalla completa** y **navegador con el panel**; o **solo el panel** en pantalla compartida si la sesión es online.
    - Orden sugerido en la app: **Comparativa** (impacto) → **Explorador** (un experimento CNN y el mejor DEIMv2) → **Visualizaciones** (2–3 imágenes que muestren claramente el fallo de la CNN y el acierto del ViT).
    - Tener **2–3 imágenes “ensayadas”** (IDs o rutas) para no perder tiempo buscando.

### Cierre (2 diapositivas)

13. **Conclusiones** (3 viñetas máximo) + **contribuciones** (dataset, comparativa, herramienta).
14. **Trabajo futuro** breve (más datos, otros dominios, VLMs) y **gracias / preguntas**.

---

## 4. Consejos para un público no experto en visión artificial

- **Definir una vez:** mAP como “resumen de qué tan bien acierta el detector en todas las clases” (sin entrar en IoU salvo que pregunten).
- **Evitar siglas en cascada** en la misma frase; si usas ViT, SSL y mAP, reparte en diapositivas distintas.
- **Analogía útil:** atención global ≈ “ver la foto entera antes de decidir”; CNN ≈ “ir ampliando trozos”.
- **Énfasis en ingeniería:** falsos positivos = paradas de línea; por eso la memoria destaca **precisión** y umbrales — conecta con **coste operativo**.
- **Tiempo:** si vas largo, recorta antecedentes teóricos y refuerza **figura de resultados + demo**.

---

## 5. Checklist previa al día de la defensa

- [ ] Streamlit probado en el **mismo PC y resolución** que usarás (o ensayo general en la sala).
- [ ] Datos en `herramienta_comparativa/data/` presentes y `experiments_metadata.json` cargando sin error.
- [ ] Modo **pantalla completa** del navegador y zoom legible (125–150 % si la proyección es lejana).
- [ ] PDF de respaldo: **3–4 capturas fijas** de la app por si falla la red o el entorno.
- [ ] Diapositiva de “Plan B”: resultados clave en una sola lámina si no hay demo.

---

## 6. Ideas opcionales para una PPT “más chula” (sin perder seriedad)

- Paleta **única** (2 colores + gris) alineada con la portada del TFG o con el panel Streamlit.
- **Una** animación simple: aparición de cajas en una imagen (antes/después) — solo si no resta tiempo.
- Iconos consistentes (industria, GPU, dataset, métrica) de un mismo pack libre.
- En la diapositiva de metodología, **línea temporal horizontal** que copie la lógica de la vista **Línea temporal** del dashboard (refuerzo visual entre PPT y demo).

---

*Documento de apoyo para preparar la defensa; ajustar tiempos y número de diapositivas según las normas de tu escuela y el tiempo asignado.*
