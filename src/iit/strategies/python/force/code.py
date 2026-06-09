"""Estrategia BruteForce: explora todas las 2^(m+n) biparticiones y retorna la de mínima EMD."""

import time

from src.iit.base.consts import FLOAT_ZERO, INFTY_POS
from src.iit.base.funcs import emd_efecto
from src.iit.core.solution import Solution
from src.iit.strategies.python.func import fmt_biparticion_fuerza_bruta
from src.iit.strategies.python.sia import SIA
from src.infra.middlewares.profile import perfilar
from src.infra.middlewares.slogger import SafeLogger

log = SafeLogger("force")


@perfilar
class BruteForce(SIA, nombre="force"):
    """Explora las 2^(m+n) biparticiones y devuelve la de mínima EMD."""

    def resolver(self) -> Solution:
        dm_original = self.distribucion
        indices = self.sistema.indices
        dims = self.sistema.dims
        m, n = len(indices), len(dims)
        full_f, full_p = (1 << m) - 1, (1 << n) - 1

        mejor_perdida = INFTY_POS
        mejor_dist = dm_original
        mejor_prim_mech: tuple = ()
        mejor_prim_purv: tuple = ()
        mejor_dual_mech: tuple = ()
        mejor_dual_purv: tuple = ()
        t0 = time.perf_counter()
        found_zero = False

        log.error(f"indices(purv)={indices} dims(mec)={dims} dm_orig={dm_original}")

        for mask_f in range(1 << m):
            if found_zero:
                break
            for mask_p in range(1 << n):
                if (mask_f == 0 and mask_p == 0) or (
                    mask_f == full_f and mask_p == full_p
                ):
                    continue
                subalcance = tuple(indices[i] for i in range(m) if mask_f >> i & 1)
                submecanismo = tuple(dims[i] for i in range(n) if mask_p >> i & 1)
                dm = self.sistema.bipartir(
                    subalcance, submecanismo
                ).distribucion_marginal()
                emd = emd_efecto(dm, dm_original)
                log.error(
                    f"  cut purv={subalcance} mec={submecanismo} → emd={emd:.6f} dm={dm}"
                )
                if emd < mejor_perdida:
                    mejor_perdida = emd
                    mejor_dist = dm
                    mejor_prim_mech = submecanismo
                    mejor_prim_purv = subalcance
                    mejor_dual_mech = tuple(d for d in dims if d not in submecanismo)
                    mejor_dual_purv = tuple(i for i in indices if i not in subalcance)
                    if emd == FLOAT_ZERO:
                        found_zero = True
                        break

        log.error(
            f"MIP: perdida={mejor_perdida:.6f} purv={mejor_prim_purv}|{mejor_dual_purv} mec={mejor_prim_mech}|{mejor_dual_mech}"
        )

        texto = fmt_biparticion_fuerza_bruta(
            [mejor_prim_mech, mejor_prim_purv],
            [mejor_dual_mech, mejor_dual_purv],
        )
        return Solution(
            estrategia=self.nombre.capitalize(),
            perdida=float(mejor_perdida) if mejor_perdida != INFTY_POS else FLOAT_ZERO,
            distribucion_subsistema=dm_original,
            distribucion_particion=mejor_dist,
            particion=texto.strip(),
            tiempo_total=time.perf_counter() - t0,
            quiere_hablar=False,
        )
