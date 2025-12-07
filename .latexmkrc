$pdflatex = 'pdflatex -synctex=1 -interaction=nonstopmode -file-line-error -shell-escape %O %S';
$pdf_mode = 1;

# Añadir carpeta 'sty' a las rutas de búsqueda (TEXINPUTS)
# El // final indica búsqueda recursiva dentro de sty
ensure_path('TEXINPUTS', './sty//');

# Función auxiliar para manejar variables de entorno de rutas de forma segura
sub ensure_path {
    my ($var, $value) = @_;
    if ($ENV{$var}) {
        $ENV{$var} = $value . $os_path_sep . $ENV{$var};
    } else {
        $ENV{$var} = $value . $os_path_sep;
    }
}

