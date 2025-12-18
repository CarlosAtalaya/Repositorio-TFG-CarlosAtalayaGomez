#!/bin/bash

# Script de compilación automática para TFG LaTeX
# Autor: Carlos Atalaya Gómez
# Descripción: Limpia, compila y procesa bibliografía del proyecto TFG

set -e  # Salir si hay algún error

# Colores para output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Función para imprimir mensajes con color
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[OK]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Verificar que estamos en el directorio correcto
if [ ! -f "TFG.tex" ]; then
    print_error "No se encuentra TFG.tex. ¿Estás en el directorio raíz del proyecto?"
    exit 1
fi

print_info "Iniciando compilación del TFG..."
echo ""

# Paso 1: Limpiar archivos auxiliares
print_info "Limpiando archivos auxiliares de LaTeX..."
rm -f *.aux *.log *.fls *.fdb_latexmk *.synctex.gz *.bbl *.bcf *.blg *.run.xml *.out *.toc *.lot *.lof *.acn *.ist *.w18 *.acr *.alg *.glg *.glo *.gls *.idx *.ilg *.ind *.lol *.nav *.nlo *.snm *.vrb 2>/dev/null
print_success "Archivos auxiliares eliminados"
echo ""

# Paso 2: Primera compilación (genera .bcf para biber)
print_info "Primera compilación (generando .bcf para bibliografía)..."
if latexmk -pdf -interaction=nonstopmode TFG.tex > /dev/null 2>&1; then
    print_success "Primera compilación completada"
else
    print_warning "Primera compilación completada con warnings (normal en primera pasada)"
fi
echo ""

# Paso 3: Procesar bibliografía con biber
if [ -f "TFG.bcf" ]; then
    print_info "Procesando bibliografía con biber..."
    if biber TFG > /dev/null 2>&1; then
        print_success "Bibliografía procesada correctamente"
    else
        print_warning "Biber completado (algunos warnings pueden ser normales)"
    fi
    echo ""
else
    print_warning "No se encontró TFG.bcf, saltando procesamiento de bibliografía"
    echo ""
fi

# Paso 4: Recompilar para incluir bibliografía
print_info "Recompilando para incluir bibliografía y resolver referencias..."
if latexmk -pdf -interaction=nonstopmode TFG.tex > /dev/null 2>&1; then
    print_success "Recompilación completada"
else
    print_warning "Recompilación completada con warnings"
fi
echo ""

# Paso 5: Compilación final forzada para resolver todas las referencias
print_info "Compilación final (resolviendo referencias cruzadas)..."
if latexmk -pdf -f -interaction=nonstopmode TFG.tex > /dev/null 2>&1; then
    print_success "Compilación final completada"
else
    print_warning "Compilación final completada (algunos warnings pueden persistir)"
fi
echo ""

# Verificar que el PDF se generó
if [ -f "TFG.pdf" ]; then
    PDF_SIZE=$(du -h TFG.pdf | cut -f1)
    print_success "PDF generado correctamente: TFG.pdf (${PDF_SIZE})"
    echo ""
    
    # Mostrar resumen de warnings (si existen)
    if [ -f "TFG.log" ]; then
        WARNINGS=$(grep -i "warning" TFG.log | wc -l)
        ERRORS=$(grep -i "error" TFG.log | grep -v "0 error" | wc -l)
        
        if [ "$WARNINGS" -gt 0 ] || [ "$ERRORS" -gt 0 ]; then
            print_warning "Se encontraron ${WARNINGS} warnings y ${ERRORS} errores en el log"
            print_info "Revisa TFG.log para más detalles"
            echo ""
        fi
    fi
    
    print_success "¡Compilación completada exitosamente!"
    print_info "Puedes abrir TFG.pdf para ver los cambios"
    
else
    print_error "No se pudo generar TFG.pdf. Revisa TFG.log para más detalles."
    exit 1
fi

