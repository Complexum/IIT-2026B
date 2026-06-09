# analytic_mpi — Concurrencia distribuida (MPI)

> La **optimalidad** del resultado está probada en [`../analytic/demo.md`](../analytic/demo.md)
> (normalización signada $\delta$, $|\cdot|$ tras promediar). Este documento cubre **solo** cómo se
> distribuye el cómputo entre ranks y por qué el resultado es idéntico al serial. Es el espejo
> multi-nodo de [`../analytic_mul/demo.md`](../analytic_mul/demo.md): misma partición de máscaras,
> distinto transporte (red MPI en vez de IPC local).

---

## 1. Qué se distribuye

Sobre las $M = 2^{D-1}$ máscaras no triviales,

$$f(m) = \sum_{i=1}^{N} \min\!\left(\left|\frac{S_h(i, m)}{2^{a}}\right|,\; \left|\frac{S_h(i, m^c)}{2^{D-a}}\right|\right), \qquad a=\operatorname{popcount}(m),\; m^c = m \oplus \texttt{full}.$$

Cada $f(m)$ depende solo de las columnas $m, m^c$ de $\texttt{sumas} \in \mathbb{R}^{N\times 2^D}$
(Zeta transform) → las $M$ evaluaciones son **independientes**. Se distribuyen entre los
$P = \texttt{comm.size} - 1$ ranks worker; el rank 0 es el driver.

---

## 2. Modelo BSP de la ejecución

La ejecución es un único superpaso *scatter → compute → gather* sobre el `MPIPoolExecutor`:

1. **Broadcast del estado (una vez).** El driver difunde $\texttt{sumas}$, $\texttt{full}$ y $D$ a
   los $P$ ranks vía `executor.map(_init_worker, [sumas]*P, …)`. Cada rank guarda $\texttt{sumas}$
   en estado global.

   $$\text{Coste de comunicación} = \Theta\!\left(P \cdot N \cdot 2^D \cdot 4\ \text{bytes}\right) \quad\text{(cuello de botella de red).}$$

2. **Scatter de máscaras.** $\{1,\ldots,M\} = \bigsqcup_{w=0}^{P-1} C_w$, bloques contiguos
   $|C_w| \approx \lceil M/P \rceil$. Solo se envían los índices `int32` de $C_w$ (despreciable).

3. **Compute local.** El rank $w$ evalúa $\{f(m): m \in C_w\}$, $\Theta(|C_w|\cdot N)$ flops, sin
   comunicación entre ranks.

4. **Gather + reduce.** El driver recoge los $P$ vectores parciales, los concatena y aplica
   $\arg\min$ (asociativo) más el desempate de EMD real. Sólo regresan $|C_w|$ floats por rank.

$$W = \Theta(N \cdot 2^D)\ \text{(igual al serial)}, \qquad T_{\text{compute}} = \Theta\!\left(\tfrac{N\cdot 2^D}{P}\right), \qquad T_{\text{comm}} = \Theta(P\,N\,2^D)\ \text{(paso 1)}.$$

El paso 1 domina la comunicación: conviene cuando el cómputo por máscara amortiza la difusión, es
decir para $D$ moderado-alto y $N$ no trivial. El término secuencial de Amdahl es la Zeta transform
($\Theta(D N 2^D)$) en el driver.

---

## 3. Aspecto técnico / despliegue

- Backend `mpi4py.futures.MPIPoolExecutor`; extra opcional `.[mpi]` (`mpi4py>=3.1`).
- **Hard-fail** en `resolver`: sin `mpi4py` o con `comm.size < 2` → `RuntimeError` (la estrategia
  sigue registrada y visible en el CLI; solo falla al ejecutarse mal lanzada).
- Lanzamiento (cluster):

  ```bash
  uv pip install -e ".[mpi]"
  mpiexec -n <P> python -m mpi4py.futures -m src.cli run execution <name>
  ```

- **Umbral.** Si $M < \texttt{MIN\_PARALLEL\_MASKS}$ ($=64$, $D<8$), se evalúa inline en el driver
  (la latencia de red no compensa). El estado se difunde solo si se entra a la rama distribuida.

---

## 4. Exactitud (equivalencia con el serial)

**Proposición.** $\texttt{analytic\_mpi}$ produce la misma partición que $\texttt{analytic}$.

*Demostración.* Idéntica a la de `analytic_mul`: cada $f(m)$ se computa íntegro en un único rank con
la misma fórmula y la misma $\texttt{sumas}$; las máscaras forman una partición disjunta de
$\{1,\ldots,M\}$; el $\arg\min$ y el desempate de EMD real corren en el driver. La suma $\sum_i$ de
cada máscara **no se trocea** entre ranks → no hay reordenamiento de redondeo float32. El resultado
es bit-a-bit el del serial salvo el orden de gather, irrelevante para $\arg\min$ (asociativo). $\square$
