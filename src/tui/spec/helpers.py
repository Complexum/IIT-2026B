"""Formateo de una especificación para el panel de inspección."""

from src.io import cnf, pbn
from src.io.fst import EspecFST, cnf_de_espec
from src.io.topologia import corte_cerrado


def nombres_de_nodo(n: int) -> list[str]:
    """`A…Z, AA, AB, …` — ver `cnf.nombre_de_variable` para el porqué del esquema."""
    return cnf.nombres_de_variables(n)


def formatear_formulas(espec: EspecFST) -> str:
    """Una línea por nodo con su fórmula CNF legible y sus padres."""
    nombres = nombres_de_nodo(espec.n)
    formulas = cnf_de_espec(espec)
    filas = []
    for i in range(espec.n):
        padres = ", ".join(nombres[p] for p in espec.padres[i]) or "—"
        filas.append(
            f"[b]{nombres[i]}[/b]({padres}) = {pbn.formatear(formulas[i + 1], nombres)}"
        )
    return "\n".join(filas)


def formatear_adyacencia(espec: EspecFST) -> str:
    """Matriz de adyacencia: fila = origen, columna = destino."""
    nombres = nombres_de_nodo(espec.n)
    ancho = max(len(x) for x in nombres)
    cabecera = " " * (ancho + 2) + " ".join(x.rjust(ancho) for x in nombres)
    filas = [cabecera]
    for origen in range(espec.n):
        celdas = [
            ("1" if origen in espec.padres[destino] else "·").rjust(ancho)
            for destino in range(espec.n)
        ]
        filas.append(f"{nombres[origen].rjust(ancho)}  " + " ".join(celdas))
    return "\n".join(filas)


def formatear_diagnostico(espec: EspecFST) -> str:
    """Fan-in, tipo de tablas y —lo importante— si la red es separable.

    Una red separable admite un corte cerrado, y ése cuesta `f = 0` sin importar
    los valores de las tablas: `φ = 0` sale trivial. Conviene saberlo al crearla y
    no después de un barrido entero.
    """
    fan_in = espec.fan_in
    tipo = "binaria (determinista)" if espec.es_binaria() else "probabilística (PBN)"
    compacto = sum(t.size for t in espec.tablas)
    materializada = espec.n * (1 << espec.n)

    hallado = corte_cerrado(espec.padres)
    if hallado is None:
        veredicto = "[b green]no separable[/b green] — no existe corte cerrado propio"
    else:
        A, alcance = hallado
        nombres = nombres_de_nodo(espec.n)
        dims = ", ".join(nombres[d] for d in sorted(A))
        veredicto = (
            f"[b red]SEPARABLE[/b red] — hay un corte cerrado con A = {{{dims}}} "
            f"y alcance de {len(alcance)} nodos, que da φ = 0 por construcción "
            f"sin importar las tablas"
        )

    return (
        f"nodos: {espec.n}   fan-in: min {min(fan_in)} / max {max(fan_in)}\n"
        f"tablas: {tipo}\n"
        f"tamaño: {compacto} valores en la spec  vs  {materializada:,} en la TPM "
        f"({materializada / max(compacto, 1):,.0f}×)\n"
        f"{veredicto}"
    )
