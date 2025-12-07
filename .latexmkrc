# Configuración de latexmk para TFG
# Habilita shell-escape (necesario para minted)
$pdflatex = 'pdflatex -synctex=1 -interaction=nonstopmode -file-line-error -shell-escape %O %S';
$pdf_mode = 1;

# Añadir carpeta 'sty' y 'tex' a las rutas de búsqueda (TEXINPUTS)
# En Linux el separador es ':'
# El // final indica búsqueda recursiva
$ENV{'TEXINPUTS'} = './sty//:./tex//:' . ($ENV{'TEXINPUTS'} // '');
