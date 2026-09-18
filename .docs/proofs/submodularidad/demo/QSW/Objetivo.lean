import Mathlib

/-!
# El objetivo de QSW como función de conjunto

Transcripción fiel de `src/iit/strategies/python/qsw/oracle.py:56-83` (`f_cara`).

El corte que busca el MIP es un subconjunto `S ⊆ V = 𝒟 ⊎ ℐ`, donde `𝒟` son las `D`
dimensiones activas (candidatas a *mecanismo*, tiempo `t`) e `ℐ` los `N` índices
(candidatos a *alcance*, tiempo `t+1`). Escribiendo `A = mecanismo S` y
`B = alcance S`, y `δᵢ = Hᵢ - pᵢ` para la normalización con signo del hipercubo `i`
contra su estado pivote, el objetivo es

  f(S) = Σ_{i ∉ B} |δ̄ᵢ(A)| + Σ_{i ∈ B} |δ̄ᵢ(Aᶜ)|

donde `δ̄ᵢ(A)` es la media de `δᵢ` sobre la cara en la que las dimensiones de `A`
corren libres y el resto queda fijo en el pivote. En el código esas medias se leen
de una tabla Zeta precomputada; acá se definen directamente, que es lo mismo.

Se trabaja sobre `ℚ` y no sobre `ℝ` a propósito: los contraejemplos son aritmética
exacta con denominador 4, y en `ℚ` cierran sin análisis.
-/

namespace QSW

variable {D N : ℕ}

/-- Una **cara** del hipercubo de dimensiones: el conjunto de dimensiones que corren
libres (las demás quedan fijas en el pivote). -/
abbrev Cara (D : ℕ) := Finset (Fin D)

/-- Los vértices del corte: `D` dimensiones (mecanismo, en `t`) más `N` índices
(alcance, en `t+1`). Es la bipartición fiel de las dos capas temporales. -/
abbrev Vertice (D N : ℕ) := Fin D ⊕ Fin N

/-- Las dimensiones que `S` pone del lado del mecanismo. -/
def mecanismo (S : Finset (Vertice D N)) : Cara D :=
  Finset.univ.filter (fun j => Sum.inl j ∈ S)

/-- Los índices que `S` pone del lado del alcance. -/
def alcance (S : Finset (Vertice D N)) : Finset (Fin N) :=
  Finset.univ.filter (fun i => Sum.inr i ∈ S)

/-- Media de `δ` sobre la cara `A`: `2^(-|A|) · Σ_{x ⊆ A} δ x`.

Es exactamente la lectura `sumas[i, m] / 2^popcount(m)` de `f_cara`, donde `sumas`
es la transformada Zeta (suma sobre subconjuntos) de `δ`. -/
def media (δ : Cara D → ℚ) (A : Cara D) : ℚ :=
  (∑ x ∈ A.powerset, δ x) / 2 ^ A.card

/-- El objetivo de QSW: el EMD del corte `S`, sumado sobre los `N` hipercubos.

Cada hipercubo aporta la media sobre `A` si su índice quedó **fuera** del alcance, y
la media sobre `Aᶜ` si quedó **dentro** — el cruce de capas que hace del corte una
bipartición y no dos elecciones independientes. -/
def f (δ : Fin N → Cara D → ℚ) (S : Finset (Vertice D N)) : ℚ :=
  ∑ i, if i ∈ alcance S then |media (δ i) (mecanismo S)ᶜ| else |media (δ i) (mecanismo S)|

/-- Submodularidad en la forma de mathlib: `g(S ∪ T) + g(S ∩ T) ≤ g(S) + g(T)`.

Es la hipótesis exacta bajo la cual el algoritmo de Queyranne garantiza el óptimo, y
por tanto lo que hay que decidir para saber si QSW da un certificado o una heurística. -/
def Submodular {α : Type*} [DecidableEq α] (g : Finset α → ℚ) : Prop :=
  ∀ S T : Finset α, g (S ∪ T) + g (S ∩ T) ≤ g S + g T

@[simp]
lemma mem_mecanismo {S : Finset (Vertice D N)} {j : Fin D} :
    j ∈ mecanismo S ↔ Sum.inl j ∈ S := by
  simp [mecanismo]

@[simp]
lemma mem_alcance {S : Finset (Vertice D N)} {i : Fin N} :
    i ∈ alcance S ↔ Sum.inr i ∈ S := by
  simp [alcance]

@[simp]
lemma mecanismo_compl (S : Finset (Vertice D N)) :
    mecanismo Sᶜ = (mecanismo S)ᶜ := by
  ext j; simp

@[simp]
lemma alcance_compl (S : Finset (Vertice D N)) :
    alcance Sᶜ = (alcance S)ᶜ := by
  ext i; simp

@[simp]
lemma mecanismo_union (S T : Finset (Vertice D N)) :
    mecanismo (S ∪ T) = mecanismo S ∪ mecanismo T := by
  ext j; simp

@[simp]
lemma mecanismo_inter (S T : Finset (Vertice D N)) :
    mecanismo (S ∩ T) = mecanismo S ∩ mecanismo T := by
  ext j; simp

@[simp]
lemma alcance_union (S T : Finset (Vertice D N)) :
    alcance (S ∪ T) = alcance S ∪ alcance T := by
  ext i; simp

@[simp]
lemma alcance_inter (S T : Finset (Vertice D N)) :
    alcance (S ∩ T) = alcance S ∩ alcance T := by
  ext i; simp

/-- La media sobre la cara vacía es el valor en el pivote: no queda nada que promediar. -/
@[simp]
lemma media_empty (δ : Cara D → ℚ) : media δ ∅ = δ ∅ := by
  simp [media]

end QSW
