"""Execution tab: crear y ejecutar programas.

Cada programa combina: dataset + patrón de prueba + estrategia.
"""

import csv
import os
import signal
import subprocess
import time
from pathlib import Path

from textual import work
from textual.app import ComposeResult
from textual.containers import Horizontal, VerticalScroll
from textual.widget import Widget
from textual.widgets import Button, Label, Rule

from src.iit.core.params import Params
from src.iit.strategies.runner import ejecutar
from src.infra.middlewares.slogger import get_logger
from src.infra.monitoring.resource_monitor import ResourceMonitor
from src.io.manager import cargar_mpt, dims_de_red
from src.tui.run.csv_utils import CSV_HEADERS, cargar_indices_completados
from src.tui.run.helpers import (
    Programa,
    build_mpi_command,
    build_output_stem,
    cargar_meta_programa,
    cargar_programa,
    eliminar_programa,
    es_estrategia_mpi,
    guardar_meta_completo,
    guardar_meta_programa,
    guardar_programa,
    listar_programas,
    siguiente_nombre_programa,
)
from src.tui.run.platform import build_plataforma
from src.tui.run.widgets import ProgramCard
from src.tui.shared.consts import estilizar
from src.tui.test.generadores import etiquetas
from src.tui.test.helpers import cargar_patron, generar_combinaciones

log = get_logger("execution")

OUTPUT_DIR = Path("data/output")


class ExecutionScreen(Widget):
    """Execution tab: gestión y ejecución de programas."""

    DEFAULT_CSS = estilizar()

    def compose(self) -> ComposeResult:
        # Toolbar
        with Horizontal(id="exec-toolbar"):
            yield Button("+ Nuevo Programa", variant="primary", id="btn-nuevo-prog")
            yield Label("Programas de ejecución", classes="toolbar-titulo")

        yield Rule()

        # Listado scrolleable de tarjetas de programa
        with VerticalScroll(id="exec-lista"):
            pass  # ProgramCards se agregan dinámicamente

    def on_mount(self) -> None:
        self._cancelando: set[str] = set()
        self._mpi_procesos: dict[str, subprocess.Popen] = {}
        self.__refrescar_programas()

    # ── Eventos ──

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn-nuevo-prog":
            self.__crear_programa()

    def on_program_card_iniciado(self, event: ProgramCard.Iniciado) -> None:
        """Empezar ejecución real de un programa en hilo de fondo."""
        self.__update_card_ejecutando(event.nombre, True)
        try:
            prog = cargar_programa(event.nombre)
        except Exception:
            prog = None
        if prog and es_estrategia_mpi(prog.estrategia):
            self._ejecutar_programa_mpi(event.nombre)
        else:
            self._ejecutar_programa(event.nombre)

    def on_program_card_cancelado(self, event: ProgramCard.Cancelado) -> None:
        """Señalar al worker que debe detenerse; feedback visual inmediato."""
        self._cancelando.add(event.nombre)
        for card in self.query(ProgramCard):
            if card.nombre_programa == event.nombre:
                card.marcar_cancelando()
                break

    def on_program_card_eliminado(self, event: ProgramCard.Eliminado) -> None:
        """Eliminar programa y su tarjeta."""
        eliminar_programa(event.nombre)
        self.__refrescar_programas()

    def on_program_card_configurado(self, event: ProgramCard.Configurado) -> None:
        """Guardar cambios de configuración del programa."""
        try:
            prog = cargar_programa(event.nombre)
            prog.dataset = event.dataset
            prog.patron = event.patron
            prog.estrategia = event.estrategia
            guardar_programa(prog)
        except Exception as e:
            log.error(
                f"Failed to update program '{event.nombre}': {type(e).__name__}: {e}"
            )

    # ── Helpers privados ──

    def __crear_programa(self) -> None:
        nombre = siguiente_nombre_programa()
        prog = Programa(nombre=nombre)
        guardar_programa(prog)
        self.__refrescar_programas()

    def refrescar_datasets(self) -> None:
        """Re-render all ProgramCards to pick up changed network list."""
        self.__refrescar_programas()

    def __refrescar_programas(self) -> None:
        lista = self.query_one("#exec-lista", VerticalScroll)
        for card in list(lista.query(ProgramCard)):
            card.remove()
        for nombre in listar_programas():
            try:
                prog = cargar_programa(nombre)
                lista.mount(ProgramCard(prog))
            except Exception as e:
                log.error(f"Failed to load program '{nombre}': {type(e).__name__}: {e}")
                continue

    @work(thread=True)
    def _ejecutar_programa(self, nombre: str) -> None:
        """Ejecutar estrategia real en hilo de fondo, escribir CSV y actualizar progreso."""
        try:
            prog = cargar_programa(nombre)
            if not prog.dataset or not prog.patron or not prog.estrategia:
                log.warn(f"Program '{nombre}' missing dataset/patron/estrategia")
                return

            tpm = cargar_mpt(prog.dataset)
            patron = cargar_patron(prog.patron)
            n_dims = tpm.shape[1]
            combis = generar_combinaciones(patron, n_dims)
            total = len(combis)

            if total == 0:
                log.warn(f"Program '{nombre}': patron produces 0 combinations")
                return

            output_stem = build_output_stem(
                nombre, prog.dataset, prog.patron, prog.estrategia
            )
            output_path = OUTPUT_DIR / f"{output_stem}.csv"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            plataforma = build_plataforma()

            # Resetear progress y combo antes de arrancar (limpia ejecuciones anteriores)
            self.app.call_from_thread(self.__update_card_progress, nombre, 0.0)
            self.app.call_from_thread(self.__update_card_combo, nombre, "")

            completados = cargar_indices_completados(output_path)
            meta = cargar_meta_programa(output_stem)
            params_actuales = {
                "dataset": prog.dataset,
                "patron": prog.patron,
                "estrategia": prog.estrategia,
            }
            checkpoint_valido = bool(completados) and meta is not None and all(
                meta.get(k) == v for k, v in params_actuales.items()
            )
            es_reanudacion = checkpoint_valido
            modo_apertura = "a" if checkpoint_valido else "w"
            if not checkpoint_valido:
                completados = set()
                guardar_meta_programa(output_stem, prog)

            prog.estado = "ejecutando"
            guardar_programa(prog)

            if es_reanudacion:
                progreso_inicial = len(completados) / total * 100
                self.app.call_from_thread(
                    self.__update_card_progress, nombre, progreso_inicial
                )
                self.app.call_from_thread(
                    self.__update_card_combo,
                    nombre,
                    f"Reanudando — {len(completados)}/{total} ya completados",
                )

            cancelado = False
            i = 0
            with output_path.open(modo_apertura, encoding="utf-8", newline="") as f:
                writer = csv.writer(f, quoting=csv.QUOTE_NONNUMERIC)
                if not es_reanudacion:
                    writer.writerow(CSV_HEADERS)

                for i, (estado, condicion, alcance, mecanismo) in enumerate(combis):
                    if nombre in self._cancelando:
                        self._cancelando.discard(nombre)
                        cancelado = True
                        break

                    if i in completados:
                        progreso = (i + 1) / total * 100
                        self.app.call_from_thread(
                            self.__update_card_progress, nombre, progreso
                        )
                        continue

                    combo_text = (
                        f"Test {i + 1}/{total}: "
                        f"({estado} | {condicion} | {alcance} | {mecanismo})"
                        f"  →  ({etiquetas(alcance)} | {etiquetas(mecanismo)})"
                    )
                    self.app.call_from_thread(
                        self.__update_card_combo, nombre, combo_text
                    )
                    try:
                        params = Params(estado, condicion, alcance, mecanismo)
                        monitor = ResourceMonitor()
                        monitor.start()
                        sol = ejecutar(tpm, params, prog.estrategia)
                        stats = monitor.stop()
                        writer.writerow(
                            [
                                i,
                                estado,
                                condicion,
                                alcance,
                                mecanismo,
                                round(float(sol.perdida), 6),
                                stats.tiempo_wall_s,
                                stats.tiempo_cpu_s,
                                stats.cpu_user_s,
                                stats.cpu_sys_s,
                                stats.mem_rss_mb,
                                stats.gpu_mem_mb,
                                sol.particion.replace("\n", "\\n"),
                                plataforma,
                            ]
                        )
                        f.flush()
                    except Exception as e:
                        log.error(
                            f"Row {i} failed in '{nombre}': {type(e).__name__}: {e}"
                        )

                    progreso = (i + 1) / total * 100
                    self.app.call_from_thread(
                        self.__update_card_progress, nombre, progreso
                    )

            guardar_meta_completo(output_stem, prog, total, cancelado, output_path)

            if not cancelado:
                prog.estado = "completado"
                prog.progreso = 100.0
                guardar_programa(prog)
                self.app.call_from_thread(self.__update_card_progress, nombre, 100.0)
                self.app.call_from_thread(
                    self.__update_card_combo, nombre, f"Completado — {total} tests"
                )
                self.app.call_from_thread(self.__refrescar_results)
            else:
                self.app.call_from_thread(
                    self.__update_card_combo,
                    nombre,
                    f"Cancelado — {i}/{total} completados",
                )
                self.app.call_from_thread(self.__refrescar_results)

            self.app.call_from_thread(self.__update_card_ejecutando, nombre, False)

        except Exception as e:
            log.error(f"Failed to execute program '{nombre}': {type(e).__name__}: {e}")
            self.app.call_from_thread(self.__update_card_ejecutando, nombre, False)

    @work(thread=True)
    def _ejecutar_programa_mpi(self, nombre: str) -> None:
        """Lanzar estrategia MPI como subproceso (mpiexec) y monitorear progreso vía CSV."""
        try:
            prog = cargar_programa(nombre)
            if not prog.dataset or not prog.patron or not prog.estrategia:
                log.warn(f"Program '{nombre}' missing dataset/patron/estrategia")
                return

            n_dims = dims_de_red(prog.dataset)
            patron = cargar_patron(prog.patron)
            combis = generar_combinaciones(patron, n_dims)
            total = len(combis)

            if total == 0:
                log.warn(f"Program '{nombre}': patron produces 0 combinations")
                return

            output_stem = build_output_stem(
                nombre, prog.dataset, prog.patron, prog.estrategia
            )
            output_path = OUTPUT_DIR / f"{output_stem}.csv"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            log_path = OUTPUT_DIR / f"{output_stem}.mpi.log"

            completados = cargar_indices_completados(output_path)
            meta = cargar_meta_programa(output_stem)
            params_actuales = {
                "dataset": prog.dataset,
                "patron": prog.patron,
                "estrategia": prog.estrategia,
            }
            checkpoint_valido = bool(completados) and meta is not None and all(
                meta.get(k) == v for k, v in params_actuales.items()
            )
            if not checkpoint_valido:
                completados = set()
                guardar_meta_programa(output_stem, prog)

            prog.estado = "ejecutando"
            guardar_programa(prog)

            progreso_inicial = len(completados) / total * 100
            self.app.call_from_thread(
                self.__update_card_progress, nombre, progreso_inicial
            )
            self.app.call_from_thread(
                self.__update_card_combo,
                nombre,
                f"Lanzando MPI ({prog.n_procs} procesos)…",
            )

            cmd = build_mpi_command(prog)
            cancelado = False
            with log_path.open("w", encoding="utf-8") as logf:
                proc = subprocess.Popen(
                    cmd,
                    stdout=logf,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                self._mpi_procesos[nombre] = proc

                self.app.call_from_thread(
                    self.__update_card_combo, nombre, f"Ejecutando: {' '.join(cmd)}"
                )

                while proc.poll() is None:
                    if nombre in self._cancelando:
                        self._cancelando.discard(nombre)
                        cancelado = True
                        self.__terminar_proceso_mpi(proc)
                        break
                    time.sleep(1.0)
                    n_completados = len(cargar_indices_completados(output_path))
                    progreso = n_completados / total * 100
                    self.app.call_from_thread(
                        self.__update_card_progress, nombre, progreso
                    )
                    self.app.call_from_thread(
                        self.__update_card_combo,
                        nombre,
                        f"MPI ({prog.n_procs} procesos) — {n_completados}/{total}",
                    )

                returncode = proc.wait()

            self._mpi_procesos.pop(nombre, None)

            n_completados = len(cargar_indices_completados(output_path))
            guardar_meta_completo(output_stem, prog, total, cancelado, output_path)

            if cancelado:
                prog.estado = "pendiente"
                combo = f"Cancelado — {n_completados}/{total} completados"
            elif returncode != 0:
                prog.estado = "error"
                combo = f"Error MPI (code {returncode}) — ver {log_path}"
            else:
                prog.estado = "completado"
                prog.progreso = 100.0
                combo = f"Completado — {total} tests (MPI)"

            guardar_programa(prog)
            self.app.call_from_thread(
                self.__update_card_progress, nombre, n_completados / total * 100
            )
            self.app.call_from_thread(self.__update_card_combo, nombre, combo)
            self.app.call_from_thread(self.__refrescar_results)
            self.app.call_from_thread(self.__update_card_ejecutando, nombre, False)

        except Exception as e:
            log.error(
                f"Failed to execute MPI program '{nombre}': {type(e).__name__}: {e}"
            )
            self._mpi_procesos.pop(nombre, None)
            self.app.call_from_thread(self.__update_card_ejecutando, nombre, False)

    def __terminar_proceso_mpi(self, proc: subprocess.Popen) -> None:
        """Termina mpiexec y sus procesos hijos (grupo de proceso)."""
        try:
            if hasattr(os, "killpg"):
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            else:
                proc.terminate()
        except ProcessLookupError:
            pass

    def __update_card_progress(self, nombre: str, progreso: float) -> None:
        for card in self.query(ProgramCard):
            if card.nombre_programa == nombre:
                card.actualizar_progreso(progreso)
                break

    def __update_card_ejecutando(self, nombre: str, en_ejecucion: bool) -> None:
        for card in self.query(ProgramCard):
            if card.nombre_programa == nombre:
                card.marcar_ejecutando(en_ejecucion)
                break

    def __refrescar_results(self) -> None:
        from src.tui.results.screen import ResultsScreen
        from src.tui.analysis.screen import AnalysisScreen

        try:
            self.app.query_one(ResultsScreen).refrescar()
        except Exception:
            pass

        try:
            self.app.query_one(AnalysisScreen).refrescar()
        except Exception:
            pass

    def __update_card_combo(self, nombre: str, texto: str) -> None:
        for card in self.query(ProgramCard):
            if card.nombre_programa == nombre:
                card.actualizar_combo(texto)
                break
