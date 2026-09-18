import QSW.Objetivo
import QSW.Basico
import QSW.Modular
import QSW.NoSubmodular

/-!
# ¿Es submodular el objetivo EMD de QSW?

No. Este es el punto de entrada de la demostración formal; ver `../README.md` para el
documento legible y `README.md` para cómo compilar.

* `QSW.Objetivo`     — definiciones: media de cara, `f`, submodularidad.
* `QSW.Basico`       — `f` es anclada y simétrica (es `prop:symmetric` del paper).
* `QSW.Modular`      — `f` **sí** es modular en el alcance: los pares índice-índice dan igualdad.
* `QSW.NoSubmodular` — el teorema principal: dos contraejemplos explícitos.
-/

/- Comprobación de que el teorema principal no depende de axiomas exóticos:
solo `propext`, `Classical.choice` y `Quot.sound`. En particular, **sin `sorryAx`**. -/
#print axioms QSW.objetivo_emd_no_es_submodular
