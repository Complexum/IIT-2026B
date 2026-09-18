import QSW.Basico

/-!
# `f` es *modular* en el alcance

Fijado el mecanismo `A`, cada hipercubo `i` aporta una de dos constantes —`|δ̄ᵢ(A)|` si
queda fuera del alcance, `|δ̄ᵢ(Aᶜ)|` si queda dentro— y nada más. Una suma de elecciones
independientes es modular, así que en el alcance la desigualdad de submodularidad se
cumple **con igualdad**, en los dos sentidos.

Esto localiza el problema: toda la no-submodularidad de `f` vive en la capa de dimensiones
y en el cruce entre capas. Es la explicación del `idx-idx 0` que mide `numerico.py` sobre
N15A (0 violaciones en 20 000 muestras entre pares de índices, contra 1790 dim-dim y
4511 cruzadas): no es una casualidad estadística, es este teorema.
-/

namespace QSW

variable {D N : ℕ}

/-- **`f` es modular en el alcance.** Si dos cortes coinciden en el mecanismo, la
desigualdad de submodularidad entre ellos es una igualdad. -/
theorem f_modular_en_alcance (δ : Fin N → Cara D → ℚ) (S T : Finset (Vertice D N))
    (h : mecanismo S = mecanismo T) :
    f δ (S ∪ T) + f δ (S ∩ T) = f δ S + f δ T := by
  have hU : mecanismo (S ∪ T) = mecanismo S := by simp [← h]
  have hI : mecanismo (S ∩ T) = mecanismo S := by simp [← h]
  unfold f
  rw [hU, hI, ← h, ← Finset.sum_add_distrib, ← Finset.sum_add_distrib]
  refine Finset.sum_congr rfl fun i _ => ?_
  simp only [alcance_union, alcance_inter, Finset.mem_union, Finset.mem_inter]
  by_cases hS : i ∈ alcance S <;> by_cases hT : i ∈ alcance T <;> simp [hS, hT, add_comm]

/-- **Corolario: las segundas diferencias sobre pares de índices se anulan.**

Mover dos índices del alcance —uno, el otro, o los dos— no produce ninguna interacción:
el efecto conjunto es exactamente la suma de los efectos por separado. Es la forma en que
`numerico.py` mide la modularidad, y da cero por este corolario. -/
theorem segunda_diferencia_indices (δ : Fin N → Cara D → ℚ) (S : Finset (Vertice D N))
    (u v : Fin N) (huv : u ≠ v)
    (hu : Sum.inr u ∉ S) (hv : Sum.inr v ∉ S) :
    f δ (insert (Sum.inr u) (insert (Sum.inr v) S)) + f δ S
      = f δ (insert (Sum.inr u) S) + f δ (insert (Sum.inr v) S) := by
  have hmec : mecanismo (insert (Sum.inr u) S) = mecanismo (insert (Sum.inr v) S) := by
    ext j; simp
  have := f_modular_en_alcance δ (insert (Sum.inr u) S) (insert (Sum.inr v) S) hmec
  rwa [show insert (Sum.inr u) S ∪ insert (Sum.inr v) S
        = insert (Sum.inr u) (insert (Sum.inr v) S) by ext x; simp; tauto,
      show insert (Sum.inr u) S ∩ insert (Sum.inr v) S = S by
        ext x
        simp only [Finset.mem_inter, Finset.mem_insert]
        constructor
        · rintro ⟨hu' | hu', hv' | hv'⟩ <;> simp_all
        · tauto] at this

end QSW
