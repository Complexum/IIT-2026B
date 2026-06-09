# Paralelización de la Estrategia Analítica IIT

## 1. Resumen Matemático del Algoritmo

### 1.1 Variables y Dominio

- **N**: Número de hipercubos (ncubos)
- **D**: Dimensión de cada hipercubo (ejes)
- **C[i]**: Hipercubo i-ésimo con forma $(2, 2, ..., 2)$ (D veces)
- **Pivote**: Posición $(0, 0, ..., 0)$ — estado inicial
- **$\nabla_i$**: Diferencias absolutas $|C_i - C_i[0,...,0]|$

### 1.2 Ecuaciones Fundamentales

#### Concentración (un ncubo absorbe todo)

$$C_{conc} = \frac{\min_{k \in [N]} S_\Omega^{(k)}}{2^D}$$

donde $S_\Omega^{(k)} = \sum_{x \in \{0,1\}^D} \nabla_k[x]$ es la suma total del hipercubo k.

#### Distribución (partición de dimensiones)

Para cada partición $(A, B)$ donde $A \cup B = \{1...D\}$, $A \cap B = \emptyset$:

$$C_{dist}(A,B) = \sum_{i=1}^{N} \min\left(\frac{S_h(i,A)}{2^{|A|}}, \frac{S_h(i,B)}{2^{|B|}}\right)$$

donde $S_h(i,A)$ es la suma de la hiper-cara fijando ejes fuera de $A$ al índice 0.

#### Decisión Final

$$\text{resultado} = \min\left(C_{conc}, \min_{(A,B) \in \mathcal{P}(D)} C_{dist}(A,B)\right)$$

### 1.3 Complejidad Original

- **Normalización**: $O(N \cdot 2^D)$
- **Concentración**: $O(N \cdot 2^D)$
- **Distribución**: $O(2^{D-1} \cdot N \cdot 2^D) = O(N \cdot 4^D)$
- **Total**: $O(N \cdot 4^D)$ para D fijo

**Número de particiones a evaluar**: $2^{D-1} - 1$

| D   | Particiones | Complejidad por partición |
| --- | ----------- | ------------------------- |
| 2   | 1           | $O(N \cdot 4)$            |
| 3   | 3           | $O(N \cdot 8)$            |
| 4   | 7           | $O(N \cdot 16)$           |
| 8   | 127         | $O(N \cdot 256)$          |

---

## 2. Análisis de Paralelización Matemática

### 2.1 Identificación de Paralelismo

El algoritmo tiene **tres niveles de paralelismo independientes**:

#### Nivel 1: Paralelismo entre N Cubos

Las operaciones sobre cada ncubo son **completamente independientes**:
- Cálculo de $\nabla_i$ (normalización)
- Cálculo de $S_\Omega^{(i)}$ (suma total)
- Cálculo de $S_h(i,A)$ para cualquier $A$

**Granularidad**: Gruesa (cada ncubo es una unidad de trabajo)
**Comunicación**: Nula entre cálculos de diferentes ncubos
**Sincronización**: Solo al final para reducir mínimos y sumas

#### Nivel 2: Paralelismo entre Particiones

Cada partición $(A,B)$ se evalúa **independientemente**:
- Para cada $A \subseteq \{1...D\}$, calculamos $C_{dist}(A,B)$
- No hay dependencias entre diferentes particiones
- El mínimo global se reduce al final

**Granularidad**: Media (cada partición requiere iterar sobre N cubos)
**Comunicación**: Nula durante evaluación
**Sincronización**: Reducción de mínimo al final

#### Nivel 3: Paralelismo Interno (Operaciones Vectoriales)

Dentro de cada ncubo, las operaciones son vectorizables:
- Suma sobre hiper-caras: reducción sobre subespacios
- Cálculo de mínimos: operaciones elemento-a-elemento
- Resta del pivote: broadcasting

**Granularidad**: Fina (operaciones sobre elementos individuales)
**Comunicación**: Compartida (misma memoria del ncubo)
**Sincronización**: Barreras implícitas en reducciones

### 2.2 Descomposición Matemática para Paralelización

#### Estrategia 1: Paralelización por N (Data Parallelism)

**Asignación**: Cada worker procesa un subconjunto de ncubos $N_p \subseteq [N]$

Para cada partición $(A,B)$:
1. Worker $w$ calcula contribución parcial: $C_{dist}^{(w)}(A,B) = \sum_{i \in N_w} \min(S_h(i,A), S_h(i,B))$
2. Reducción global: $C_{dist}(A,B) = \sum_{w} C_{dist}^{(w)}(A,B)$

**Ventaja**: Perfecto balanceo de carga si N es grande
**Desventaja**: Sobrecarga si D es grande (cada worker hace mucho trabajo)

#### Estrategia 2: Paralelización por Particiones (Task Parallelism)

**Asignación**: Cada worker procesa un subconjunto de particiones $\mathcal{P}_w \subseteq \mathcal{P}(D)$

Para cada worker $w$:
1. Iterar sobre $(A,B) \in \mathcal{P}_w$
2. Calcular $C_{dist}(A,B)$ completo (todos los N cubos)
3. Mantener mínimo local
4. Reducción global de mínimos

**Ventaja**: Mínima comunicación durante cálculo
**Desventaja**: Posible desbalanceo si $|A|$ varía (trabajo por partición depende de $|A|$)

#### Estrategia 3: Paralelización Híbrida (N + Particiones)

**Asignación 2D**: Grid de workers $(w_N, w_P)$

- $w_N$ se encarga de un subconjunto de ncubos
- $w_P$ se encarga de un subconjunto de particiones
- Cada worker calcula una submatriz de contribuciones

**Ventaja**: Máximo paralelismo, escala con ambos N y D
**Desventaja**: Mayor complejidad de implementación y comunicación

---

## 3. Técnicas de Implementación por Plataforma

### 3.1 Mac M4 Pro (Apple Silicon)

#### Características del Hardware

- **CPU**: Apple M4 Pro (14 cores: 10 performance + 4 efficiency)
- **GPU**: Integrada (18-20 cores GPU, arquitectura Apple)
- **Neural Engine**: 16-core NPU (dedicado a ML)
- **Memoria**: Unified Memory Architecture (RAM compartida CPU/GPU)
- **Sin CUDA**: No NVIDIA GPU disponible

#### Opción A: Multiprocessing Python (CPU)

**Tecnología**: `multiprocessing` o `concurrent.futures.ProcessPoolExecutor`

**Aplicación**:
- **Nivel 1 (Paralelismo por N)**: Ideal para N grande
  ```python
  def evaluar_ncubos_batch(ncubos_batch, particiones):
      resultados = {}
      for A, B in particiones:
          costo = sum(min(suma_hipercara(c, A), suma_hipercara(c, B)) 
                     for c in ncubos_batch)
          resultados[(A,B)] = costo
      return resultados
  
  # Pool de 10 procesos (performance cores)
  with ProcessPoolExecutor(max_workers=10) as executor:
      futures = [executor.submit(evaluar_ncubos_batch, batch, particiones) 
                for batch in dividir_ncubos(data, 10)]
  ```

- **Nivel 2 (Paralelismo por Particiones)**: Ideal para D grande
  ```python
  def evaluar_particion(A, B, ncubos):
      return sum(min(suma_hipercara(c, A), suma_hipercara(c, B)) 
                for c in ncubos)
  
  with ProcessPoolExecutor(max_workers=10) as executor:
      futures = [executor.submit(evaluar_particion, A, B, ncubos) 
                for A, B in particiones]
  ```

**Ventajas**:
- Sin GIL (Global Interpreter Lock)
- Escalable hasta 10 cores de performance
- Fácil implementación en Python puro

**Limitaciones**:
- Overhead de serialización de arrays NumPy
- Memoria duplicada entre procesos (aunque UMA ayuda)
- No aprovecha GPU

#### Opción B: Parallel NumPy/SciPy (CPU Vectorizado)

**Tecnología**: NumPy con OpenBLAS/MKL (ya paraleliza operaciones matriciales)

**Optimizaciones**:
- Configurar `OPENBLAS_NUM_THREADS=10`
- Usar operaciones matriciales en lugar de loops Python
- Vectorizar cálculos sobre todos los ncubos simultáneamente

```python
# Vectorización completa (sin loops Python)
data_nd = np.stack([c.ndata for c in ncubos])  # (N, 2, 2, ..., 2)

# Todas las normalizaciones simultáneas
pivot_vals = data_nd[(slice(None),) + (0,)*D]  # (N,)
diffs = np.abs(data_nd - pivot_vals.reshape(N, *([1]*D)))

# Suma total de cada ncubo
S_omega = diffs.reshape(N, -1).sum(axis=1)  # (N,)
C_conc = S_omega.min() / (2**D)
```

**Ventajas**:
- Código limpio y mantenible
- Aprovecha SIMD y múltiples cores automáticamente
- Sin overhead de multiprocessing

**Limitaciones**:
- Aún limitado por CPU
- Para D grande, la memoria puede ser problema

#### Opción C: Metal Performance Shaders (GPU)

**Tecnología**: `pytorch` con backend MPS o `mlx` (Apple ML framework)

**Implementación con MLX**:
```python
import mlx.core as mx

# Transferir datos a GPU (unified memory, casi instantáneo)
data_mx = mx.array(data_nd)  # (N, 2, 2, ..., 2)

# Operaciones en GPU
pivot = data_mx[:, 0, 0, ..., 0]  # (N,)
diffs = mx.abs(data_mx - pivot.reshape(N, *([1]*D)))

# Reducciones en GPU
S_omega = mx.sum(diffs.reshape(N, -1), axis=1)
C_conc = mx.min(S_omega) / (2**D)
```

**Implementación con PyTorch MPS**:
```python
import torch

device = torch.device("mps")
data_t = torch.tensor(data_nd, device=device)

# Operaciones en GPU Apple
pivot = data_t[(slice(None),) + (0,)*D]
diffs = torch.abs(data_t - pivot.view(N, *([1]*D)))
S_omega = diffs.view(N, -1).sum(dim=1)
C_conc = S_omega.min() / (2**D)
```

**Ventajas**:
- Aprovecha 18-20 GPU cores
- Unified Memory: cero copia entre CPU y GPU
- MLX optimizado específicamente para Apple Silicon

**Limitaciones**:
- GPU Apple no es tan potente como NVIDIA para compute general
- MLX es nuevo (2024), menos maduro que CUDA
- PyTorch MPS tiene algunas operaciones no soportadas

#### Opción D: Neural Engine (NPU)

**Tecnología**: CoreML o CreateML

**Aplicación**:
- La NPU está optimizada para operaciones matriciales estáticas
- Útil si podemos expresar el problema como una red neuronal
- Overhead de compilación solo vale la pena para ejecuciones repetidas

**Ventajas**:
- 16 cores dedicados
- Muy eficiente energéticamente
- Ideal para batch processing

**Limitaciones**:
- Requiere convertir el problema a formato CoreML
- Menos flexible que CPU/GPU
- Solo útil si el patrón de acceso es regular

### 3.2 PC con NVIDIA GPU (CUDA)

#### Características Típicas

- **CPU**: Multi-core x86 (Intel/AMD)
- **GPU**: NVIDIA (RTX 4090, A100, etc.)
- **CUDA Cores**: Miles de cores
- **Memoria**: GPU dedicada (VRAM) + RAM del sistema

#### Opción A: CUDA con Numba

**Tecnología**: `numba.cuda`

**Implementación**:
```python
from numba import cuda
import numpy as np

@cuda.jit
def evaluar_particion_kernel(data, pivot_idx, A_mask, B_mask, N, D, resultado):
    """Cada thread evalúa un ncubo para una partición."""
    idx = cuda.grid(1)
    if idx < N:
        # Calcular S_h(i, A) y S_h(i, B)
        sum_A = 0.0
        sum_B = 0.0
        # Iterar sobre todos los elementos del hipercubo
        for flat_idx in range(2**D):
            # Convertir índice plano a coordenadas D-dimensionales
            coords = []
            temp = flat_idx
            for d in range(D):
                coords.append(temp % 2)
                temp //= 2
            
            # Verificar si está en hiper-cara A o B
            in_A = all(coords[d] == 0 for d in range(D) if not A_mask[d])
            in_B = all(coords[d] == 0 for d in range(D) if not B_mask[d])
            
            if in_A:
                sum_A += data[idx, flat_idx]
            if in_B:
                sum_B += data[idx, flat_idx]
        
        # Calcular mínimo
        val_A = sum_A / (2**sum(A_mask))
        val_B = sum_B / (2**sum(B_mask))
        resultado[idx] = min(val_A, val_B)

# Lanzar kernel
threads_per_block = 256
blocks = (N + threads_per_block - 1) // threads_per_block
evaluar_particion_kernel[blocks, threads_per_block](
    d_data, d_pivot, d_A_mask, d_B_mask, N, D, d_resultado
)
# Reducción en GPU o CPU
C_dist = d_resultado.sum()
```

**Ventajas**:
- Control total sobre paralelización
- Máximo rendimiento en GPU NVIDIA
- Python-friendly

**Limitaciones**:
- Complejidad de implementación CUDA
- Debugging más difícil
- Portabilidad limitada

#### Opción B: CuPy (NumPy en GPU)

**Tecnología**: `cupy`

**Implementación**:
```python
import cupy as cp

# Transferir a GPU
data_cp = cp.array(data_nd)  # (N, 2, 2, ..., 2)
pivot_cp = data_cp[(slice(None),) + (0,)*D]
diffs_cp = cp.abs(data_cp - pivot_cp.reshape(N, *([1]*D)))

# Para cada partición
for A, B in particiones:
    # Crear máscaras para hiper-caras
    mask_A = cp.ones((2,)*D, dtype=bool)
    mask_B = cp.ones((2,)*D, dtype=bool)
    for d in range(D):
        if d not in A:
            mask_A[(slice(None),)*d + (0,) + (slice(None),)*(D-d-1)] = False
    
    # Aplicar máscara y sumar
    S_h_A = (diffs_cp * mask_A).reshape(N, -1).sum(axis=1)
    S_h_B = (diffs_cp * mask_B).reshape(N, -1).sum(axis=1)
    
    costo = cp.minimum(S_h_A / (2**len(A)), S_h_B / (2**len(B))).sum()
    C_dist = min(C_dist, costo)
```

**Ventajas**:
- API casi idéntica a NumPy
- Fácil migración desde código CPU
- Buen rendimiento sin complejidad de CUDA crudo

**Limitaciones**:
- Menos control que CUDA directo
- Algunas operaciones avanzadas no soportadas
- Overhead de transferencia CPU-GPU

#### Opción C: PyTorch CUDA

**Tecnología**: `torch.cuda`

**Implementación**:
```python
import torch

device = torch.device("cuda")
data_t = torch.tensor(data_nd, device=device)

# Similar a MPS pero en CUDA
pivot = data_t[(slice(None),) + (0,)*D]
diffs = torch.abs(data_t - pivot.view(N, *([1]*D)))

# Optimización: precalcular todas las hiper-caras posibles
# Para D=8, hay 256 hiper-caras posibles (2^D)
# Pero solo necesitamos las relevantes para las particiones
```

**Ventajas**:
- Ecosistema maduro
- Autograd disponible (si se necesita diferenciación)
- Fácil profiling y optimización

**Limitaciones**:
- Overhead de PyTorch para problema simple
- Memoria GPU puede limitar N y D

#### Opción D: MPI para Multi-GPU / Multi-Nodo

**Tecnología**: `mpi4py`

**Aplicación**:
- Si tienes múltiples GPUs o un cluster
- Distribuir N cubos o particiones entre nodos
- Comunicación eficiente vía MPI

```python
from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Distribuir ncubos entre procesos MPI
ncubos_por_proceso = N // size
inicio = rank * ncubos_por_proceso
fin = inicio + ncubos_por_proceso if rank < size - 1 else N

# Cada proceso evalúa su subset
mi_data = data_nd[inicio:fin]
mi_resultado = evaluar_particiones(mi_data, particiones)

# Reducción global
total = comm.reduce(mi_resultado, op=MPI.SUM, root=0)
```

**Ventajas**:
- Escala a múltiples máquinas
- Muy eficiente para N grande
- Estándar en HPC

**Limitaciones**:
- Setup más complejo
- Overhead de comunicación entre nodos
- Requiere infraestructura de cluster

### 3.3 Comparativa de Técnicas

| Técnica           | Hardware   | Paralelismo       | N Ideal    | D Ideal | Complejidad | Rendimiento |
| ----------------- | ---------- | ----------------- | ---------- | ------- | ----------- | ----------- |
| Multiprocessing   | CPU (M4)   | N o Particiones   | Grande     | Medio   | Baja        | Medio       |
| NumPy Vectorizado | CPU (M4)   | SIMD/Threads      | Medio      | Medio   | Muy Baja    | Medio-Alto  |
| MLX/PyTorch MPS   | GPU Apple  | Todos los niveles | Grande     | Grande  | Media       | Alto        |
| Numba CUDA        | GPU NVIDIA | Todos los niveles | Grande     | Grande  | Alta        | Muy Alto    |
| CuPy              | GPU NVIDIA | Vectorial         | Grande     | Grande  | Baja        | Alto        |
| MPI + CUDA        | Multi-GPU  | Todos los niveles | Muy Grande | Grande  | Muy Alta    | Extremo     |

---

## 4. Estrategia Recomendada por Escenario

### Escenario 1: N pequeño, D pequeño (N<10, D<4)
- **Solución**: NumPy vectorizado en CPU (M4)
- **Razón**: Overhead de paralelización no vale la pena
- **Tiempo esperado**: < 1ms

### Escenario 2: N grande, D pequeño (N>1000, D<6)
- **Solución**: Multiprocessing (M4) o CuPy (PC CUDA)
- **Razón**: Paralelismo por N es muy efectivo
- **Speedup esperado**: 5-10x en M4, 50-100x en CUDA

### Escenario 3: N pequeño, D grande (N<100, D=8)
- **Solución**: MLX/MPS (M4) o CUDA (PC)
- **Razón**: 127 particiones, cada una requiere mucho cómputo
- **Speedup esperado**: 10-20x en M4, 100-500x en CUDA

### Escenario 4: N grande, D grande (N>1000, D=8)
- **Solución**: Híbrido - MPI + CUDA
- **Razón**: Máximo paralelismo en ambas dimensiones
- **Speedup esperado**: 1000x+ en cluster multi-GPU

---

## 5. Optimizaciones Avanzadas

### 5.1 Precálculo de Hiper-Caras

Para D fijo, el número de hiper-caras posibles es $2^D$ (cada dimensión puede estar fija o libre).

**Idea**: Precalcular todas las $2^D$ sumas posibles para cada ncubo.

```python
# Precálculo O(N * 2^D * D)
sumas_hipercaras = np.zeros((N, 2**D))
for i in range(N):
    for mask in range(2**D):
        # mask indica qué dimensiones están fijas
        sumas_hipercaras[i, mask] = calcular_suma_con_mascara(diffs[i], mask)

# Evaluación de particiones O(num_particiones * N)
for A, B in particiones:
    mask_A = mask_de_particion(A)
    mask_B = mask_de_particion(B)
    costo = np.minimum(
        sumas_hipercaras[:, mask_A] / (2**len(A)),
        sumas_hipercaras[:, mask_B] / (2**len(B))
    ).sum()
```

**Speedup**: Reduce factor D en complejidad de evaluación

### 5.2 Memoización de Particiones

Si se evalúan múltiples sistemas con la misma D:
- Las particiones y sus máscaras se reutilizan
- Precalcular todas las combinaciones `combinations(range(D), k)`

### 5.3 Early Exit

Si en medio de la evaluación de una partición la suma parcial ya supera el mejor C_dist encontrado:
```python
for A, B in particiones:
    costo = 0
    for i in range(N):
        costo += min(S_h_A[i], S_h_B[i])
        if costo >= best_dist:
            break  # Early exit
```

### 5.4 Compresión de Datos

Los valores son distancias absolutas en [0, 1] (después de normalización):
- Posible usar float16 en GPU para mayor throughput
- O int8 si la precision permite

---

## 6. Pseudocódigo Paralelo Completo

```python
# Configuración
PLATAFORMA = "M4"  # o "CUDA"
PARALELISMO = "HIBRIDO"  # "N", "PARTICIONES", "HIBRIDO"

# Carga de datos
data_nd = cargar_ncubos()  # (N, 2, 2, ..., 2)
N, D = data_nd.shape[0], data_nd.ndim - 1

if PLATAFORMA == "M4":
    if PARALELISMO == "N":
        # Multiprocessing por N
        from multiprocessing import Pool
        n_workers = 10  # performance cores
        batches = dividir(data_nd, n_workers)
        
        with Pool(n_workers) as pool:
            resultados = pool.starmap(evaluar_todas_particiones, 
                                     [(batch, particiones) for batch in batches])
        
        # Reducir resultados
        C_conc = min(r["C_conc"] for r in resultados)
        C_dist = min(r["C_dist"] for r in resultados)
        
    elif PARALELISMO == "HIBRIDO":
        # MLX para GPU + CPU
        import mlx.core as mx
        data_mx = mx.array(data_nd)
        # ... (ver sección 3.1 Opción C)
        
else:  # CUDA
    import cupy as cp
    data_cp = cp.array(data_nd)
    # ... (ver sección 3.2 Opción B)

resultado = min(C_conc, C_dist)
```

---

## 7. Métricas de Evaluación

Para medir el éxito de la paralelización:

1. **Speedup**: $S(p) = \frac{T_1}{T_p}$
2. **Eficiencia**: $E(p) = \frac{S(p)}{p}$
3. **Throughput**: N cubos evaluados por segundo
4. **Memoria**: Pico de uso de memoria (CPU/GPU)

**Objetivos**:
- Speedup lineal hasta 10 cores (M4) o 1000+ cores (CUDA)
- Eficiencia > 80% para N > 1000
- Overhead de paralelización < 5% del tiempo total

---

## 8. Conclusión

La estrategia analítica es **inherentemente paralelizable** en múltiples niveles:

1. **Nivel Matemático**: Las sumas sobre ncubos y particiones son independientes
2. **Nivel Algorítmico**: Data parallelism y task parallelism coexisten
3. **Nivel Hardware**: CPU (multiprocessing), GPU (CUDA/Metal), y NPU son todos viables

**Para Mac M4 Pro**:
- **Recomendación primaria**: MLX o PyTorch MPS para GPU
- **Recomendación secundaria**: NumPy vectorizado con OpenBLAS paralelo
- **Evitar**: Multiprocessing puro (overhead innecesario)

**Para PC con CUDA**:
- **Recomendación primaria**: CuPy (fácil) o Numba CUDA (máximo rendimiento)
- **Recomendación secundaria**: PyTorch CUDA (si ya se usa en el proyecto)
- **Para producción**: MPI + CUDA multi-GPU

**Próximos pasos**:
1. Implementar versión NumPy optimizada (baseline)
2. Benchmark en M4 con diferentes N y D
3. Implementar versión MLX para GPU Apple
4. Migrar a CuPy/Numba para PC CUDA
5. Comparar resultados y elegir la mejor estrategia híbrida
