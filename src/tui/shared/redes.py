"""Refresco cruzado entre tabs cuando cambia el estado de las redes en disco.

Una red es un nombre con hasta tres archivos (spec/csv/npy) y tres tabs la
muestran con sus marcas Ⓢ🅒Ⓝ. Sin este módulo, borrar un CSV desde Dataset
dejaba a Specs mostrando una marca 🅒 fantasma — cada tab sólo refrescaba su
propia lista en sus propios eventos.

Convención: todo tab que muestra redes implementa `refrescar_lista()`; todo tab
que muta el disco llama a `refrescar_tabs_de_redes()` después.
"""

from textual.app import App


def refrescar_tabs_de_redes(app: App) -> None:
    """Refresca la lista de redes de todos los tabs que la muestran.

    Los imports son locales: dataset ↔ spec se importan mutuamente y un import
    top-level crearía un ciclo.
    """
    from src.tui.dataset.screen import DatasetScreen
    from src.tui.run.screen import ExecutionScreen
    from src.tui.spec.screen import SpecScreen

    for cls in (DatasetScreen, SpecScreen, ExecutionScreen):
        try:
            app.query_one(cls).refrescar_lista()
        except Exception:  # noqa: BLE001 — tab no montado todavía
            pass
