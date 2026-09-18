import QSW.Objetivo

/-!
# `f` es una función de corte legítima: anclada y simétrica

Es la Proposición `prop:symmetric` del paper (`04-methods.tex:283`), mecanizada.

No sorprende a nadie —está probada a mano y testeada en
`tests/strategies/test_qsw.py:test_f_es_simetrica_y_anclada`— pero cumple un papel
concreto acá: si las definiciones de `Objetivo.lean` estuvieran mal transcritas,
estas dos pruebas fallarían. Pasan, así que el contraejemplo de `NoSubmodular.lean`
habla del objetivo de verdad y no de una función parecida.
-/

namespace QSW

variable {D N : ℕ}

@[simp]
lemma mecanismo_empty : mecanismo (∅ : Finset (Vertice D N)) = ∅ := by
  ext j; simp

@[simp]
lemma alcance_empty : alcance (∅ : Finset (Vertice D N)) = ∅ := by
  ext i; simp

@[simp]
lemma mecanismo_univ : mecanismo (Finset.univ : Finset (Vertice D N)) = Finset.univ := by
  ext j; simp

@[simp]
lemma alcance_univ : alcance (Finset.univ : Finset (Vertice D N)) = Finset.univ := by
  ext i; simp

/-- **`f` está anclada en el corte vacío.** La hipótesis `δ i ∅ = 0` es la
normalización con signo contra el pivote: `δᵢ = Hᵢ - pᵢ` se anula justamente ahí. -/
theorem f_empty (δ : Fin N → Cara D → ℚ) (h : ∀ i, δ i ∅ = 0) :
    f δ ∅ = 0 := by
  simp [f, h]

/-- **`f` está anclada en el corte total.** Con todo del mismo lado no se pierde nada,
igual que con nada. -/
theorem f_univ (δ : Fin N → Cara D → ℚ) (h : ∀ i, δ i ∅ = 0) :
    f δ Finset.univ = 0 := by
  simp [f, h]

/-- **`f` es simétrica:** `f Sᶜ = f S`. Complementar el corte intercambia las dos caras
candidatas y a la vez las dos ramas del condicional; los dos intercambios se cancelan.

Junto con `f_empty` y `f_univ`, esto pone a `f` exactamente en la clase de funciones que
minimiza el algoritmo de Queyranne — de ahí que la pregunta por la submodularidad sea la
única que queda abierta, y la que decide `NoSubmodular.lean`. -/
theorem f_compl (δ : Fin N → Cara D → ℚ) (S : Finset (Vertice D N)) :
    f δ Sᶜ = f δ S := by
  unfold f
  simp only [mecanismo_compl, alcance_compl, compl_compl]
  refine Finset.sum_congr rfl fun i _ => ?_
  by_cases h : i ∈ alcance S <;> simp [h]

end QSW
