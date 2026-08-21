# QSW — notas algorítmico

## 1. El problema

MIP = bipartición que minimiza pérdida de información. Enumeración exhaustiva = $2^{D-1}$ biparticiones. Inviable pasando pocos nodos.

Conjunto de vértices: $V = D + N$.

- D dimensiones activas (tiempo t) → candidatas a mecanismo
- N índices (tiempo t+1) → candidatos a alcance/purview

Un corte $S \subseteq V$ = entero de V bits. Máscara → `(alcance, mecanismo)` = un shift y un and (`code.py:99-106`). `qn` en cambio paga `__flatten`/`sorted` recursivo por consulta.

## 2. El oráculo (coste mayoritario)

$$f(S) = \sum_i \big|\text{media}_{\delta_i}(F_i)\big|$$

con $\delta_i = H_i - p_i$ (hipercubo menos su valor en el estado pivote), y $F_i$ = cara del mecanismo o su complemento según si i está en el alcance (main.tex eq. 8, `oracle.py:f_cara`).

Todas las $2^D$ sumas de cara salen de una transformada Zeta (suma-sobre-subconjuntos, patrón butterfly, `zeta.py:zeta_inplace`). Después, cada consulta es lectura de tabla, $O(N)$.

Costo: $\Theta(D \cdot N \cdot 2^D)$. Piso duro: $\Omega(N \cdot 2^D)$ — es el tamaño de la entrada. Ningún método exacto puede ser polinómico en D.
<!-- Revisar posibles representaciones alternas -->

## 3. Por qué f es un corte legítimo

| Propiedad                | Por qué                                                                                   |
| ------------------------ | ----------------------------------------------------------------------------------------- |
| $f(\emptyset) = 0$       | mecanismo vacío → $\delta(p) = 0$                                                         |
| $f(V) = 0$               | complemento vacío → todos los nodos caen en el alcance                                    |
| $f(S) = f(V\setminus S)$ | complementar intercambia `val_a ↔ val_b` y la rama del `where`; los dos swaps se cancelan |

Simétrica + anclada = exactamente la clase de función que Queyranne minimiza, y de la que el corte de grafo (Stoer-Wagner) es el caso particular. Test: `test_f_es_simetrica_y_anclada`.

## 4. Dónde entra Stoer-Wagner — el puente (esto es el corazón de la charla)

Queyranne construye un legal ordering eligiendo

$$v_{i+1} = \arg\min_v\; \big[f(A\cup\{v\}) - f(\{v\})\big]$$

Stoer-Wagner elige el vértice de máxima adyacencia $w(A,v)$ al conjunto actual.

Para una f de corte de grafo: $f(A\cup v) = f(A) + f(v) - 2\,w(A,v)$. Sustituyendo:

$$f(A\cup v) - f(v) = f(A) - 2\,w(A,v)$$

$f(A)$ no depende de v. Entonces minimizar el lado izquierdo $\equiv$ maximizar $w(A,v)$.

> **Stoer-Wagner ES Queyranne restringido a funciones gráficas. Son el mismo algoritmo.**

Consecuencia operativa — y es toda la ganancia:

- Queyranne consulta el oráculo dentro del bucle de ordenamiento: $O(V^2)$ por fase, $O(V^3)$ total.
- Stoer-Wagner no consulta nada: mantiene matriz densa W y actualiza `key[w] += W[t][w]` en O(1) (`core.py:mao_phase`).

Los dos últimos del orden (s,t) — el par colgante: el corte que aísla t es candidata de Queyranne. Después se contrae $s \leftarrow s\cup t$ y arranca otra fase. $V-1$ fases.

## 5. El surrogate de segundo orden

f no es gráfica, así que W no viene dada — se ajusta:

$$W[u][v] = \tfrac{1}{2}\big(f(\{u\}) + f(\{v\}) - f(\{u,v\})\big)$$

Si f fuera gráfica esto recupera el peso exacto de la arista: $f(\{u\}) = \deg(u)$, $f(\{u,v\}) = \deg(u)+\deg(v)-2w_{uv}$, los grados se cancelan (`test_seed_recupera_los_pesos_del_grafo`).

Si no lo es → puede dar entradas negativas; el max-adjacency solo ordena, así que sigue funcionando.

Punto clave para defender: **W únicamente guía.** Toda candidata se re-puntúa con la f exacta, y la ganadora se reconstruye con `bipartir + distribucion_marginal + emd_efecto` reales (`code.py:217`) — ningún error puede propagarse al resultado reportado.

Costo: seed $O(V^2)$ consultas + refrescar solo la fila del supernodo tras cada contracción, $\sum V_p = O(V^2)$. Total $O(V^2)$ en consultas.

## 6. Estructura completa del algoritmo

```
1. SEED        W (surrogate) + singletons                  1 batch
2. FASES       V−1 × [ MAO con keys incrementales
                       → par colgante (s,t)
                       → candidata = miembros[t]
                       → contracción ]
3. RE-SCORING  todas las candidatas con f EXACTA           1 batch
4. 1-OPT       vecinos de TODAS las candidatas             1 batch
5. 2-OPT       |C|·C(V,2) máscaras, con tope               1 batch
6. RECONSTRUIR el ganador con EMD real
```

Dos modos de contracción (`modo=`):

| modo               | tras contraer                                  | batches      | trade-off                                                         |
| ------------------ | ---------------------------------------------- | ------------ | ----------------------------------------------------------------- |
| `exacto` (default) | recalcula la fila del supernodo con el oráculo | $O(V)$ (≈29) | sin drift                                                         |
| `estatico`         | `W[s] += W[t]` (SW puro)                       | 1            | 100 % paralelizable; el error de orden se acumula por contracción |

## 7. Las optimizaciones, una por una

|    # | Técnica                                                                                                                                                                                                                                                                          | Ganancia                                                          |
| ---: | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------- |
|    1 | Puente SW×Queyranne: MAO sin oráculo                                                                                                                                                                                                                                             | $O(V^3) \to O(V^2)$ consultas                                     |
|    2 | Batching: las consultas se sirven en $O(V)$ llamadas numpy, no un round-trip Python→numpy por consulta                                                                                                                                                                           | La ganancia real: `qn` a V=30 hace ~1.2k round-trips, QSW hace 29 |
|    3 | Pivot folding (`zeta.py`): pliega las celdas relativo al pivote. El gather era el 95 % del Zeta y asignaba 738 MB de índices a D=22. Se pliega en la dirección del butterfly: la cara m vive en la tabla sin expandir; el pivote se aplica al índice de consulta, no a los datos | 1.3×                                                              |
|    4 | Kernel C bloqueado (`backend=c`): fusiona las pasadas en un recorrido con los intermedios en registros. Bit-exacto                                                                                                                                                               | 5–10×                                                             |
|    5 | `preparar_ncubos` compartidos                                                                                                                                                                                                                                                    | 3× en la reducción de subsistema                                  |
|    6 | `qsw_mul` (multiprocessing): sobrescribe solo `_preparar_oraculo`, hereda la búsqueda ⇒ resultado idéntico por construcción                                                                                                                                                      | ver §8                                                            |
|    7 | `modo=estocastico`: muestrea las caras, $O(D \cdot N \cdot 2^D) \to O(N \cdot K \cdot D^2)$                                                                                                                                                                                      | solo paga con D grande                                            |

Descartados, y esto es un resultado medido: radix-4 en numpy = 0 % (el slicing strided recorre el arreglo igual), float32 desde el origen = 4 %.

### El diagnóstico que ordena todo: memory-bandwidth bound

El precómputo sostiene 17–19 GB/s haciendo una suma por elemento. No espera cómputo, espera datos. De ahí se predice:

- quitar el gather = quitar una indirección al arreglo → paga (1.3×)
- radix-4 = no reduce pasadas ni bytes movidos → no paga (exactamente 0)
- kernel C bloqueado = fusiona pasadas → paga
- más cores en la misma máquina = no paga, comparten el mismo bus

Ese último es el resultado negativo del paper, y es una predicción cumplida, no una decepción.

> ⚠️ **Matiz que sí tienes medido** y documentado en el paper (`QSW.md` §6, `benchmarks/escalado_mul.py`): a n=20 el escalado se aplana en ~2×, pero a 29 GB/s efectivos estás en el 11 % del pico del M4 Pro. Pesan más el reparto por filas (N=20 sobre 16 workers deja rebanadas de 2 y de 1) y los 4 cores de eficiencia, no la saturación de DRAM. La conclusión operativa (hacen falta memorias separadas: GPU/HBM, nodos MPI) sobrevive; la causa exacta no. Si te preguntan, esta es la respuesta honesta.

## 8. Cómo se asegura la optimalidad — la pregunta difícil

Contesta en este orden, y empieza por lo que no tienes:

### 8.1 No hay certificado teórico, y el paper lo dice

Queyranne garantiza el óptimo cuando la función es submodular. La pérdida basada en EMD que usas es simétrica y anclada, pero no se demostró submodular. Por lo tanto:

> Lo que provee el algoritmo es una estrategia de búsqueda, no un certificado.

Está escrito explícito en `03-background.tex:26-30`. Decirlo tú primero es la posición fuerte.

### 8.2 Lo que sí está garantizado

1. **El surrogate no contamina.** W únicamente guía: toda candidata se re-puntúa con f exacta y la ganadora se reconstruye con marginalización y EMD reales. El φ del CSV nunca sale del surrogate.
2. **La máscara ganadora nunca es trivial** (ni $\emptyset$ ni V) — ambas dan $f = 0$ y se filtran (`core.py:211`).
3. **Las variantes paralelas son idénticas**, no "aproximadamente iguales". `qsw_mul` sobrescribe solo el precómputo; la búsqueda se hereda verbatim. Confirmado en las 768 instancias.
4. **El modo estocástico se auto-fianza:** si el margen entre la ganadora y la segunda no supera $2\sigma$ del ruido de muestreo, se rehace con el Zeta exacto (`muestreo.py:confianza`). Con N20A da 96/96 exacto. El diagnóstico resultó 100 % preciso (9/9): cuando dice "confiable", acierta.

### 8.3 Lo que cierra las brechas del heurístico

La búsqueda sola no alcanzaba el óptimo siempre. Dos reparaciones lo cerraron, y las dos son datos, no intuición:

- **1-opt sobre TODAS las candidatas**, no solo la ganadora. En N5B, en 3 de las 5 instancias falladas el óptimo estaba a un movimiento de una candidata que no había ganado el re-scoring — mirando sus vecinos se visitaba. Cuesta $V \cdot |C|$ máscaras en un batch $= O(V^2)$, dentro del presupuesto que la búsqueda ya paga.
- **2-opt para las 2 restantes.** En N5B[36] el 1-opt se queda en 0.109449 contra 0.037566 del óptimo, y ningún movimiento simple cierra esa brecha. Topado en `MAX_MASCARAS_2OPT = 200_000` porque crece $O(V^4)$.

Con las dos: 384/384 = 100 %.

### 8.4 La evidencia empírica

|             | vs PyPhi (`phi`) |
| ----------- | ---------------- |
| `queyranne` | 383/384 (99.7 %) |
| `qsw`       | 384/384 (100 %)  |
| `qsw_mul`   | 384/384 (100 %)  |

768 instancias, 8 redes de 5 a 20 nodos, entre ellas circuitos neuronales reales de *Drosophila melanogaster* a 15 nodos. `qsw` y `qsw_mul` dan pérdidas idénticas en las 768.

La tolerancia importa y es defendible. A 1e-6, seis instancias de N10B aparecen como desacuerdo con desviación de exactamente 1.00e-06: es la resolución de la acumulación float32 del oráculo, no una partición distinta. Los desacuerdos reales están 3–5 órdenes más arriba (2.3e-04 el de `queyranne`; 7.2e-02 el de `qsw` que motivó el 2-opt). A 1e-5 las dos poblaciones se separan limpio. A 1e-6 los tres métodos daban 98.2 % y quedaban mezclados dos fallos aritméticos con uno algorítmico.

### 8.5 El experimento que aísla la causa

Si alguien duda del oráculo (Zeta, coordenadas delta): `analytic` es exhaustivo sobre EL MISMO oráculo y coincide con PyPhi 96/96 en N5B con $\max|\Delta| = 0$. Si el precómputo estuviera roto, `analytic` fallaría igual. Es el experimento decisivo y es barato.

## 9. Los números

| Red  | `queyranne` | `qsw` |    `analytic` |
| ---- | ----------: | ----: | ------------: |
| N5A  |       0.023 | 0.039 |         0.303 |
| N10A |       0.052 | 0.098 |       111.842 |
| N15A |       0.374 | 0.344 |             — |
| N20A |      21.152 | 2.570 | (interpolado) |

El cruce en n=15 es exactamente la firma de cambiar $O(V^3)$ por $O(V^2)$: penalidad de factor constante con V chico (construir la tabla surrogate), tasa de crecimiento menor después. Dilo así — que `queyranne` gane a n=5 y n=10 confirma el análisis en vez de contradecirlo.

## 10. Lo que el O(V³) NO dice — cuenta consultas, no tiempo

- $O(V^3)$ y $O(V^2)$ cuentan **llamadas al oráculo**, no tiempo.
- A D=22: la búsqueda son 2.8 ms; el precómputo, ~1 s. Cuando la cota polinómica aplica, el precómputo exponencial ya dominó.
- Piso $\Omega(N \cdot 2^D)$ = la propia entrada.
- Lo que la búsqueda polinómica reemplaza es el término exponencial de la enumeración (las $2^{D-1}$ biparticiones), no el de leer la entrada. Por eso `qsw` le gana a `analytic` por un factor constante y no por uno exponencial — sus términos asintóticos de búsqueda difieren enormemente.
- Nota fina: el 2-opt agrega $O(V^4)$ consultas y el total ya no es $O(V^2)$ — solo `generar_candidatas` lo es, que es la propiedad que aporta el híbrido. El test `test_conteo_de_oraculo_es_cuadratico` mide `generar_candidatas` sola, deliberadamente.
