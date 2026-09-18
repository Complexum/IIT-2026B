import QSW.Modular

/-!
# El objetivo EMD de QSW **no** es submodular

El teorema principal. Se exhiben dos contraejemplos explícitos con `D = 2` dimensiones
`{a, b}` y `N = 1` hipercubo `z`, los más pequeños posibles:

* `δxor` — el nodo XOR. Viola submodularidad en un par **dimensión-dimensión**.
* `δcruz` — viola en un par **dimensión-índice**, el cruce entre capas temporales.

Los dos son realizables por TPMs válidas (`δ(x) = p(pivote) − p(x)` con todas las
probabilidades en `[0,1]`), así que no son artefactos del formalismo: son redes.

La consecuencia está dicha en el paper y ahora está probada: lo que da el algoritmo de
Queyranne sobre este objetivo es una **estrategia de búsqueda, no un certificado**.
-/

namespace QSW

/-! ## Las cuatro caras de `Cara 2`

`media δ A` promedia `δ` sobre los subconjuntos de `A`. Con `D = 2` hay cuatro caras, y
estas cuatro igualdades son la tabla que usa el resto del archivo. -/

lemma media_vacia (δ : Cara 2 → ℚ) : media δ ∅ = δ ∅ := by simp

lemma media_a (δ : Cara 2 → ℚ) : media δ {0} = (δ ∅ + δ {0}) / 2 := by
  rw [media, show (({0} : Cara 2)).powerset = {∅, {0}} from by decide,
    Finset.sum_insert (by decide), Finset.sum_singleton]
  norm_num

lemma media_b (δ : Cara 2 → ℚ) : media δ {1} = (δ ∅ + δ {1}) / 2 := by
  rw [media, show (({1} : Cara 2)).powerset = {∅, {1}} from by decide,
    Finset.sum_insert (by decide), Finset.sum_singleton]
  norm_num

lemma media_ab (δ : Cara 2 → ℚ) :
    media δ Finset.univ = (δ ∅ + δ {0} + δ {1} + δ Finset.univ) / 4 := by
  rw [media, show (Finset.univ : Cara 2).powerset = {∅, {0}, {1}, Finset.univ} from by decide,
    Finset.sum_insert (by decide), Finset.sum_insert (by decide),
    Finset.sum_insert (by decide), Finset.sum_singleton]
  norm_num
  ring

/-! ## El objetivo con un solo hipercubo

Con `N = 1` la suma de `f` tiene un único término, y `f` se reduce a elegir una de las dos
caras según de qué lado del corte quedó el índice `z`. -/

/-- El vértice índice: el único hipercubo, en `t+1`. -/
abbrev vz : Vertice 2 1 := Sum.inr 0

lemma f_uno (δ : Fin 1 → Cara 2 → ℚ) (S : Finset (Vertice 2 1)) :
    f δ S = if vz ∈ S then |media (δ 0) (mecanismo S)ᶜ| else |media (δ 0) (mecanismo S)| := by
  simp [f]

/-! ## Contraejemplo 1 — el nodo XOR, par dimensión-dimensión

`δxor` vale `1` en la cara completa y `0` en las demás: es la interacción de orden 2 pura.
Sale de la TPM `p(z=1 | ab) = (1, 1, 1, 0)` con pivote `00`, vía `δ(x) = p(00) − p(x)`.

La tabla completa de `f` sobre los 8 subconjuntos de `V = {a, b, z}` — los mismos números
que imprime `numerico.py` evaluando `f_cara` de `oracle.py`:

| `S`    | `∅` | `{a}` | `{b}` | `{a,b}` | `{z}` | `{a,z}` | `{b,z}` | `{a,b,z}` |
|--------|-----|-------|-------|---------|-------|---------|---------|-----------|
| `f(S)` | 0   | 0     | 0     | ¼       | ¼     | 0       | 0       | 0         | -/

def δxor : Fin 1 → Cara 2 → ℚ := fun _ x => if x = Finset.univ then 1 else 0

@[simp] lemma δxor_vacia : δxor 0 ∅ = 0 := by rw [δxor]; exact if_neg (by decide)
@[simp] lemma δxor_a : δxor 0 {0} = 0 := by rw [δxor]; exact if_neg (by decide)
@[simp] lemma δxor_b : δxor 0 {1} = 0 := by rw [δxor]; exact if_neg (by decide)
@[simp] lemma δxor_ab : δxor 0 Finset.univ = 1 := by rw [δxor]; exact if_pos rfl

example : f δxor ∅ = 0 := by
  rw [f_uno, if_neg (by decide), show mecanismo (∅ : Finset (Vertice 2 1)) = ∅ from by decide]
  simp

example : f δxor {Sum.inl 0} = 0 := by
  rw [f_uno, if_neg (by decide),
    show mecanismo ({Sum.inl 0} : Finset (Vertice 2 1)) = {0} from by decide, media_a]
  norm_num

example : f δxor {Sum.inl 1} = 0 := by
  rw [f_uno, if_neg (by decide),
    show mecanismo ({Sum.inl 1} : Finset (Vertice 2 1)) = {1} from by decide, media_b]
  norm_num

example : f δxor {Sum.inl 0, Sum.inl 1} = 1 / 4 := by
  rw [f_uno, if_neg (by decide),
    show mecanismo ({Sum.inl 0, Sum.inl 1} : Finset (Vertice 2 1)) = Finset.univ from by decide,
    media_ab]
  norm_num

example : f δxor {vz} = 1 / 4 := by
  rw [f_uno, if_pos (by decide),
    show (mecanismo ({vz} : Finset (Vertice 2 1)))ᶜ = Finset.univ from by decide, media_ab]
  norm_num

/-- **`f` no es submodular.** Con `S = {a}`, `T = {b}`:
`f(S ∪ T) + f(S ∩ T) = ¼ > 0 = f(S) + f(T)`.

La violación es de orden 2 pura: cada dimensión por separado no pierde nada, las dos juntas
pierden ¼. Justo el retorno **creciente** que la submodularidad prohíbe. -/
theorem xor_no_es_submodular : ¬ Submodular (f δxor) := by
  intro h
  have := h {Sum.inl 0} {Sum.inl 1}
  rw [show ({Sum.inl 0} ∪ {Sum.inl 1} : Finset (Vertice 2 1)) = {Sum.inl 0, Sum.inl 1} from by
        decide,
    show ({Sum.inl 0} ∩ {Sum.inl 1} : Finset (Vertice 2 1)) = ∅ from by decide] at this
  rw [f_uno, if_neg (by decide),
      show mecanismo ({Sum.inl 0, Sum.inl 1} : Finset (Vertice 2 1)) = Finset.univ from by decide,
      media_ab] at this
  rw [f_uno, if_neg (by decide),
      show mecanismo (∅ : Finset (Vertice 2 1)) = ∅ from by decide] at this
  rw [f_uno, if_neg (by decide),
      show mecanismo ({Sum.inl 0} : Finset (Vertice 2 1)) = {0} from by decide, media_a] at this
  rw [f_uno, if_neg (by decide),
      show mecanismo ({Sum.inl 1} : Finset (Vertice 2 1)) = {1} from by decide, media_b] at this
  norm_num at this

/-! ## Contraejemplo 2 — par dimensión-índice, el cruce entre capas

`δcruz` vale `½` en la cara `{b}` y `−½` en la cara completa. Sale de la TPM
`p(z=1 | ab) = (½, ½, 0, 1)` con pivote `00`.

Esta es la violación que el algoritmo encuentra más seguido en redes reales: sobre N15A,
4511 de las 6301 violaciones muestreadas son de este tipo (ver `numerico.py`). No es un
caso de borde — mover un índice de lado cambia **qué cara** se lee, y con ella el signo del
retorno de cada dimensión. -/

def δcruz : Fin 1 → Cara 2 → ℚ :=
  fun _ x => if x = {1} then 1 / 2 else if x = Finset.univ then -(1 / 2) else 0

@[simp] lemma δcruz_vacia : δcruz 0 ∅ = 0 := by
  rw [δcruz]; rw [if_neg (by decide), if_neg (by decide)]
@[simp] lemma δcruz_a : δcruz 0 {0} = 0 := by
  rw [δcruz]; rw [if_neg (by decide), if_neg (by decide)]
@[simp] lemma δcruz_b : δcruz 0 {1} = 1 / 2 := by rw [δcruz]; rw [if_pos rfl]
@[simp] lemma δcruz_ab : δcruz 0 Finset.univ = -(1 / 2) := by
  rw [δcruz]; rw [if_neg (by decide), if_pos rfl]

/-- **La violación también aparece entre capas.** Con `S = {a}` (una dimensión) y `T = {z}`
(el índice): `f(S ∪ T) + f(S ∩ T) = ¼ > 0 = f(S) + f(T)`.

Importa que sea un par cruzado: dice que la no-submodularidad no se arregla tratando las
dos capas por separado. Y contrasta con `segunda_diferencia_indices` de `Modular.lean`,
donde los pares índice-índice sí dan igualdad exacta. -/
theorem cruzado_no_es_submodular : ¬ Submodular (f δcruz) := by
  intro h
  have := h {Sum.inl 0} {vz}
  rw [show ({Sum.inl 0} ∪ {vz} : Finset (Vertice 2 1)) = {Sum.inl 0, vz} from by decide,
    show ({Sum.inl 0} ∩ {vz} : Finset (Vertice 2 1)) = ∅ from by decide] at this
  rw [f_uno, if_pos (by decide),
      show (mecanismo ({Sum.inl 0, vz} : Finset (Vertice 2 1)))ᶜ = {1} from by decide,
      media_b] at this
  rw [f_uno, if_neg (by decide),
      show mecanismo (∅ : Finset (Vertice 2 1)) = ∅ from by decide] at this
  rw [f_uno, if_neg (by decide),
      show mecanismo ({Sum.inl 0} : Finset (Vertice 2 1)) = {0} from by decide, media_a] at this
  rw [f_uno, if_pos (by decide),
      show (mecanismo ({vz} : Finset (Vertice 2 1)))ᶜ = Finset.univ from by decide,
      media_ab] at this
  norm_num at this

/-- **El teorema, en la forma en que lo usa el paper:** no es cierto que el objetivo EMD sea
submodular para toda red. Por tanto la garantía de optimalidad de Queyranne no aplica, y las
pasadas de reparación 1-opt/2-opt de `core.py` no son un parche sino la consecuencia. -/
theorem objetivo_emd_no_es_submodular :
    ¬ ∀ (D N : ℕ) (δ : Fin N → Cara D → ℚ), Submodular (f δ) :=
  fun h => xor_no_es_submodular (h 2 1 δxor)

end QSW
