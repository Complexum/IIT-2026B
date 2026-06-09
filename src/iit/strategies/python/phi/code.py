"""Estrategia Phi: adapter a pyphi (fuerza bruta); recibe subsistema ya preparado y TPM."""

import math
import time

import numpy as np

from src.infra.middlewares.slogger import SafeLogger
from src.iit.core.solution import Solution
from src.iit.strategies.consts import DUMMY_ARR, DUMMY_PARTITION
from src.iit.strategies.python.func import ABECEDARY
from src.iit.strategies.python.sia import SIA
from src.iit.base.enums.emd import TimeEMD
from src.iit.base.app import aplicacion
from src.iit.strategies.python.func import (
    fmt_biparticion_fuerza_bruta,
    lil_endian,
)
# from src.infra.middlewares.profile import perfilar

log = SafeLogger("phi")

__PYPHI_ERR: Exception | None = None
try:
    import pyphi as _pyphi_mod
    from pyphi import Network, Subsystem
    from pyphi.labels import NodeLabels
    from pyphi.models.cuts import Bipartition, Part

    _pyphi_mod.config.VALIDATE_SUBSYSTEM_STATES = False
except Exception as __e:
    _pyphi_mod = Network = Subsystem = NodeLabels = Bipartition = Part = None
    __PYPHI_ERR = __e


# @perfilar
class Phi(SIA, nombre="phi", necesita_mpt=True):
    """Adapter a pyphi: recibe subsistema (System) y mpt; en resolver() usa pyphi para MIP."""

    def __init__(self, subsistema, tpm: np.ndarray, params=None) -> None:
        super().__init__(subsistema)
        self.mpt = tpm
        self.params = params

    def resolver(self) -> Solution:
        if Subsystem is None or Network is None:
            raise RuntimeError(f"pyphi no disponible: {__PYPHI_ERR}") from __PYPHI_ERR
        t0 = time.perf_counter()

        # Build candidato: full system conditioned on background nodes.
        # Passing the conditioned TPM to pyphi avoids StateUnreachableError
        # and ensures mechanism/purview live in the correct reduced system.
        from src.io.manager import crear_sistema

        estado = (
            self.params.estado_tuple()
            if self.params is not None
            else self.sistema.estado_inicial
        )
        dims_condicionadas = (
            self.params.dims_condicionar() if self.params is not None else ()
        )
        sistema_completo = crear_sistema(self.mpt, estado)
        candidato = sistema_completo.condicionar(dims_condicionadas)

        cand_ncubos = sorted(candidato.ncubos, key=lambda c: c.indice)
        n_hipercubos = len(cand_ncubos)
        cand_tpm = np.stack([c.data for c in cand_ncubos], axis=1).astype(np.float64)

        global_a_local = {c.indice: i for i, c in enumerate(cand_ncubos)}
        local_a_global = {i: c.indice for i, c in enumerate(cand_ncubos)}

        estado_candidato = tuple(estado[c.indice] for c in cand_ncubos)
        etiquetas = tuple(ABECEDARY[:n_hipercubos])
        red = Network(
            tpm=cand_tpm, node_labels=NodeLabels(etiquetas, tuple(range(n_hipercubos)))
        )
        subsistema_pyphi = Subsystem(
            network=red, state=estado_candidato, nodes=list(range(n_hipercubos))
        )
        mecanismo = tuple(
            global_a_local[g] for g in self.sistema.dims if g in global_a_local
        )
        alcance = tuple(
            global_a_local[g] for g in self.sistema.indices if g in global_a_local
        )
        use_effect = aplicacion.tiempo_emd == TimeEMD.EMD_EFECTO

        mip = (
            subsistema_pyphi.effect_mip(mecanismo, alcance)
            if use_effect
            else subsistema_pyphi.cause_mip(mecanismo, alcance)
        )

        small_phi: float = float(mip.phi)
        repertorio = repertorio_partido = np.array(DUMMY_ARR, dtype=np.float64)
        format_str = DUMMY_PARTITION

        if mip.repertoire is not None:
            repertorio = mip.repertoire.flatten()
            repertorio_partido = mip.partitioned_repertoire.flatten()

            states = int(math.log2(mip.repertoire.size))
            sub_estados: np.ndarray = lil_endian(states)
            repertorio.put(sub_estados, repertorio)
            repertorio_partido.put(sub_estados, repertorio_partido)
            if mip.partition is not None:
                mejor_biparticion: Bipartition = mip.partition
                prim: Part = mejor_biparticion.parts[True]
                dual: Part = mejor_biparticion.parts[False]
                prim_mech = tuple(local_a_global[i] for i in prim.mechanism)
                prim_purv = tuple(local_a_global[i] for i in prim.purview)
                dual_mech = tuple(local_a_global[i] for i in dual.mechanism)
                dual_purv = tuple(local_a_global[i] for i in dual.purview)
                format_str = fmt_biparticion_fuerza_bruta(
                    [dual_mech, dual_purv],
                    [prim_mech, prim_purv],
                )

        tiempo = time.perf_counter() - t0
        return Solution(
            estrategia=self.nombre.capitalize(),
            perdida=small_phi,
            distribucion_subsistema=repertorio,
            distribucion_particion=repertorio_partido,
            particion=format_str,
            tiempo_total=tiempo,
            quiere_hablar=False,
        )
