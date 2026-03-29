# Repositorio: memoria TFG + artículo IEEE (Vision Transformers, inspección industrial)

Este repositorio concentra la **redacción y el material fuente** asociados a un Trabajo Fin de Grado sobre **Vision Transformers y detección de defectos en componentes industriales**, y al **artículo científico** derivado (formato IEEE, carpeta dedicada). El hilo de investigación incluye comparación sistemática entre arquitecturas (por ejemplo ViT/DEIMv2 frente a CNN) y uso de contexto experimental reproducible.

## Contenido principal

| Elemento | Descripción |
|----------|-------------|
| **`TFG_Carlos_Atalaya_25-26/`** | **Memoria del TFG en LaTeX**, versión cerrada y alineada con lo entregado al tribunal. Incluye `TFG.tex`, capítulos (`tex/`), clase y estilos (`sty/`), bibliografía (`bib/`) y figuras (`fig/`). |
| **`IEEE_TII_Vision_Transformers/`** | **Paper en plantilla IEEE** (inglés), pensado para seguir desarrollándose como publicación independiente de la memoria. |
| **`Documentacion-externa/`** | **Contexto experimental**: datos (JSON, CSV), análisis, figuras auxiliares y herramientas que apoyan cifras y narrativa. Incluye `notas-trabajo-historico/` con notas y borradores de trabajo no oficiales. |

No se versionan **scripts de compilación** ni el **PDF generado** de la memoria en la raíz del repositorio; el PDF compilado localmente puede ignorarse mediante `.gitignore` (`TFG_Carlos_Atalaya_25-26/TFG.pdf`). Las figuras en formato PDF dentro de `fig/` o `sty/` sí forman parte del material fuente.

## Cómo compilar la memoria (opcional)

Abrir o sincronizar la carpeta `TFG_Carlos_Atalaya_25-26/` en un editor LaTeX (por ejemplo Overleaf) y compilar el fichero principal `TFG.tex`, o usar un flujo local habitual (`latexmk`, `pdflatex` + `biber`, etc.) según la configuración de tu máquina. Este repositorio no fija una herramienta concreta.

## Documentación de apoyo

- **`CONTEXTUALIZACION-REPOSITORIO.txt`**: descripción detallada de estructura, convenciones y uso por parte de agentes o asistentes de IA.
- **`.cursorrules`** y **`.cursor/rules/tfg_writing.mdc`**: reglas de estilo y rutas para edición asistida del LaTeX del TFG.

## Convenciones breves

- **Memoria**: español (España), formal; citas solo desde `TFG_Carlos_Atalaya_25-26/bib/main.bib`.
- **Paper**: inglés, formato IEEE según los ficheros de `IEEE_TII_Vision_Transformers/`.
- **Datos**: priorizar `Documentacion-externa/` y el texto ya consolidado en la memoria para no introducir cifras sin soporte.

---

*Estructura actualizada: marzo de 2026.*
