# Demostración formal en Lean 4

Comprueba mecánicamente lo que dice `../README.md`. Cuatro archivos:

| Archivo | Qué prueba |
|---|---|
| `QSW/Objetivo.lean` | Definiciones: cara, media de cara, `f`, submodularidad. Transcripción de `oracle.py:56-83`. |
| `QSW/Basico.lean` | `f` es anclada (`f ∅ = f univ = 0`) y simétrica (`f Sᶜ = f S`). Es `prop:symmetric` del paper. |
| `QSW/Modular.lean` | `f` **sí** es modular en el alcance: los pares índice-índice dan igualdad exacta. |
| `QSW/NoSubmodular.lean` | **El teorema principal**: `f` no es submodular. Dos contraejemplos explícitos. |

El teorema que cita el paper es `QSW.objetivo_emd_no_es_submodular`.

## Compilar

Requiere [elan](https://github.com/leanprover/elan). La versión de Lean la fija
`lean-toolchain` (`v4.32.1`) y elan la instala sola.

```sh
lake exe cache get   # mathlib precompilado: ~7 GB, necesita red
lake build
```

`lake exe cache get` es un **prerrequisito de red**, no un optimizador: sin él, `lake build`
compila mathlib entero desde cero (horas). Si falla, revisar la conexión antes que otra cosa.

`.lake/` está en `.gitignore`: nada de esos 7 GB entra al repositorio.

## Verificar que la prueba es real

`lake build` imprime, desde `QSW.lean`:

```
'QSW.objetivo_emd_no_es_submodular' depends on axioms: [propext, Classical.choice, Quot.sound]
```

Son los tres axiomas estándar de Lean. Que **no** aparezca `sorryAx` es lo que hay que mirar:
significa que no queda ningún hueco en la demostración. Comprobación adicional:

```sh
grep -rn "sorry\|native_decide" QSW/ QSW.lean   # no debe encontrar nada
```

(`native_decide` se evita a propósito: añadiría el compilador de Lean como axioma de confianza.)

## Por qué ℚ y no ℝ

Los contraejemplos son aritmética exacta con denominador 4. En ℚ cierran por `norm_num` sin
tocar análisis, y `f` queda computable. La no-submodularidad en ℚ implica la no-submodularidad
en ℝ: es la misma función, y ℚ ⊂ ℝ.
