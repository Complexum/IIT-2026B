# analytic_mul — Concurrencia por procesos (multiprocessing)

> La **optimalidad** del resultado está probada en [`../analytic/demo.md`](../analytic/demo.md)
> (normalización signada $\delta$, $|\cdot|$ tras promediar). Este documento cubre **solo** cómo se
> paraleliza el cómputo y por qué el resultado es idéntico al serial.

---

## 1. Qué se paraleliza

El costo de la estrategia serial es, sobre las $M = 2^{D-1}$ máscaras no triviales:

$$f(m) = \sum_{i=1}^{N} \min\!\left(\left|\frac{S_h(i, m)}{2^{a}}\right|,\; \left|\frac{S_h(i, m^c)}{2^{D-a}}\right|\right), \qquad a=\operatorname{popcount}(m),\; m^c = m \oplus \texttt{full}$$

con $\texttt{full} = 2^D - 1$. El ganador es $m^\star = \arg\min_m f(m)$ (más el colapso $\min_k |S_\Omega^{(k)}/2^D|$).

**Observación clave (independencia).** $f(m)$ depende solo de las columnas $m$ y $m^c$ de la tabla
precomputada $\texttt{sumas} \in \mathbb{R}^{N \times 2^D}$ (Zeta transform). Por tanto

$$\{f(m)\}_{m=1}^{M} \quad\text{son } M \text{ evaluaciones mutuamente independientes,}$$

sin dependencias de datos entre sí: paralelismo *embarrassingly parallel*.

---

## 2. Partición del trabajo

El driver (rank único) ejecuta dos fases **secuenciales** (la 2 depende de la 1):

1. **Zeta transform** $\texttt{sumas} = \text{hyperfaces}(\delta)$ — $\Theta(D \cdot N \cdot 2^D)$, secuencial.
2. **Evaluación de máscaras** — repartida en $P$ procesos.

Se particiona el conjunto de máscaras $\{1, \ldots, M\}$ en $P$ bloques contiguos casi iguales:

$$\{1,\ldots,M\} = \bigsqcup_{w=0}^{P-1} C_w, \qquad |C_w| \approx \left\lceil \tfrac{M}{P} \right\rceil, \qquad P = \min(\texttt{n\_workers}, M)$$

El worker $w$ calcula $\{f(m) : m \in C_w\}$ leyendo $\texttt{sumas}$ de su estado global
(poblado una vez por el `initializer` del `Pool`). El driver concatena los $P$ vectores parciales y
toma $\arg\min$ — operación asociativa, así que el orden de concatenación no altera el resultado.

$$W_{\text{eval}} = \Theta(N \cdot 2^D) \;\text{(igual al serial)}, \qquad T_{\text{eval}} = \Theta\!\left(\frac{N \cdot 2^D}{P}\right) + O(M)\,_{\text{argmin}}$$

Speedup ideal $\approx P$ en la fase 2; la fase 1 (Zeta) queda como término secuneial de Amdahl.

---

## 3. Modelo de coste (memoria / IPC)

`multiprocessing` es stdlib (sin dependencia extra). El estado compartido es $\texttt{sumas}$,
de tamaño $N \cdot 2^D \cdot 4$ bytes (float32). Dos regímenes según `start method`:

| método | plataforma | coste de difundir `sumas` |
| ------ | ---------- | ------------------------- |
| `fork`  | macOS / Linux | **copy-on-write**: 0 copias hasta escritura (sólo lectura → gratis) |
| `spawn` | Windows       | serializa `sumas` **una vez por worker** vía `initargs` ($P \cdot N \cdot 2^D \cdot 4$ B) |

Los chunks $C_w$ enviados a cada tarea son índices `int32` ($|C_w|$ enteros), despreciables.
La salida por tarea es un vector de $|C_w|$ float — también pequeña.

**Umbral.** Si $M < \texttt{MIN\_PARALLEL\_MASKS}$ ($= 64$, i.e. $D < 8$) o $\texttt{n\_workers} \le 1$,
el coste de fork/serialización supera la ganancia → se cae al `winner` serial óptimo (`super().winner`).

---

## 4. Exactitud (equivalencia con el serial)

**Proposición.** $\texttt{analytic\_mul}$ produce la misma partición que $\texttt{analytic}$, salvo
empates resueltos por el desempate determinista de EMD real (idéntico en ambos).

*Demostración.* Cada worker calcula $f(m)$ con la **misma** expresión numérica que el serial
(misma `sumas`, misma fórmula `min(|sA|,|sB|)`), sobre máscaras disjuntas que cubren $\{1,\ldots,M\}$.
La unión de las salidas es exactamente $\{f(m)\}_{m=1}^M$. El $\arg\min$ y el desempate
($\min$ de EMD entre los dos candidatos $m^\star, m^{\star c}$) corren en el driver, idénticos al
serial. No hay reducción en coma flotante distinta: cada $f(m)$ se computa íntegro en un solo worker
(no se trocea la suma $\sum_i$), luego no hay reordenamiento de redondeo. $\square$
