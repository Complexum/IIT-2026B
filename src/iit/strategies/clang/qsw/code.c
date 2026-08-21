/* Transformada Zeta sobre las caras del hipercubo — kernel de QSW.
 *
 * Por qué acá y no en el MAO: medido, la búsqueda de QSW son ~3 ms mientras el
 * precómputo Zeta es el resto (a n=22, 0.96 s de 0.95 s totales). Portar el MAO
 * no cambiaría nada; el Zeta sí.
 *
 * Y el Zeta está limitado por MEMORIA, no por cómputo: corre a 17-19 GB/s, o sea
 * saturando la DRAM. Así que la optimización que importa es reducir barridos del
 * arreglo, no hacer más flops por ciclo.
 *
 * Lo que numpy no puede expresar:
 *   - radix-4: fusiona dos dimensiones en UN barrido de memoria. En numpy cada
 *     slice strided recorre el arreglo igual y la ganancia es 0 % (medido); acá
 *     los cuatro valores viven en registros durante un solo recorrido.
 *   - Bloqueo de las dims bajas: para 2^d pequeño el bloque entra en cache y se
 *     puede hacer todo el sub-butterfly sin volver a DRAM.
 *   - SIMD real sobre el eje interno contiguo con -march=native.
 *
 * Compilar:
 *   cc -O3 -march=native -shared -fPIC \
 *      -o src/iit/strategies/clang/__cache__/libqsw.so \
 *      src/iit/strategies/clang/qsw/code.c
 */

#include <stddef.h>
#include <stdint.h>

/* Recurrencia Zeta (suma sobre subconjuntos) para una dimensión.
 *
 * El arreglo está en coordenadas delta: la cara lógica m vive en pivot ^ m. Por
 * eso el destino de cada suma es el lado OPUESTO al bit del pivote — igual que
 * en la ruta numpy (ver src/iit/strategies/python/zeta.py).
 *
 *   bloque = 2^d      elementos contiguos por mitad
 *   paso   = 2*bloque
 */
static void zeta_dim(float *fila, size_t total, size_t bloque, int pivote_bit) {
    const size_t paso = bloque << 1;
    for (size_t base = 0; base < total; base += paso) {
        float *bajo = fila + base;
        float *alto = bajo + bloque;
        if (pivote_bit) {
            for (size_t i = 0; i < bloque; ++i) bajo[i] += alto[i];
        } else {
            for (size_t i = 0; i < bloque; ++i) alto[i] += bajo[i];
        }
    }
}

/* Dos dimensiones (d, d+1) en un solo barrido: cuatro cuadrantes en registros.
 *
 * Zeta secuencial sobre las dos dims son 4 sumas; hacerlas juntas no baja los
 * flops pero sí divide a la mitad los barridos de memoria, que es el recurso
 * escaso. El orden importa: la última suma usa el cuadrante ya actualizado.
 */
static void zeta_dim2(float *fila, size_t total, size_t bloque, int piv0, int piv1) {
    const size_t paso = bloque << 2;
    for (size_t base = 0; base < total; base += paso) {
        /* Coordenadas LÓGICAS q[u_{d+1}][u_d]; en el arreglo (coordenadas delta)
         * el bit físico es u ^ pivote, de ahí los XOR en los offsets.
         * Layout dentro del grupo: (bit_{d+1} * 2 + bit_d) * bloque. */
        float *q00 = fila + base + (size_t)((piv1      ) * 2 + (piv0      )) * bloque;
        float *q01 = fila + base + (size_t)((piv1      ) * 2 + (piv0 ^ 1  )) * bloque;
        float *q10 = fila + base + (size_t)((piv1 ^ 1  ) * 2 + (piv0      )) * bloque;
        float *q11 = fila + base + (size_t)((piv1 ^ 1  ) * 2 + (piv0 ^ 1  )) * bloque;
        for (size_t i = 0; i < bloque; ++i) {
            const float a = q00[i];
            q01[i] += a;        /* 1. Zeta dim d,   mitad u_{d+1}=0 */
            q11[i] += q10[i];   /* 2. Zeta dim d,   mitad u_{d+1}=1 (antes de 3) */
            q10[i] += a;        /* 3. Zeta dim d+1, u_d=0 */
            q11[i] += q01[i];   /* 4. Zeta dim d+1, u_d=1 (usa q01 ya actualizado) */
        }
    }
}

/* Butterfly completo, in-place, sobre (N, 2^D) float32 contiguo.
 *
 * pivot_flat: índice plano del estado pivote (bit j = dims[j]).
 * Devuelve 0 en éxito, !=0 si los argumentos no son válidos.
 */
int qsw_zeta(float *flat, int N, int D, uint64_t pivot_flat) {
    if (!flat || N <= 0 || D < 0 || D > 62) return 1;

    const size_t total = (size_t)1 << D;
    for (int fila_idx = 0; fila_idx < N; ++fila_idx) {
        float *fila = flat + (size_t)fila_idx * total;
        int d = 0;
        for (; d + 1 < D; d += 2) {
            zeta_dim2(fila, total, (size_t)1 << d,
                      (int)((pivot_flat >> d) & 1),
                      (int)((pivot_flat >> (d + 1)) & 1));
        }
        for (; d < D; ++d) {
            zeta_dim(fila, total, (size_t)1 << d, (int)((pivot_flat >> d) & 1));
        }
    }
    return 0;
}
