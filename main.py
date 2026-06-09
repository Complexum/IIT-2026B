from src.iit.core.params import Params
from src.iit.strategies.runner import ejecutar
from src.io.manager import cargar_mpt, listar_redes


def main():
    # List available networks
    redes = listar_redes()
    print(f"Networks available: {redes}")

    if not redes:
        print("No networks found in data/input/networks/")
        return

    # Params: estado_inicial, condicion, alcance, mecanismo (bits: 0 = condicionar/quitar, 1 = mantener)
    config = Params(
        "10000",
        "11111",
        "11111",
        "11111",
    )
    N = len(config.estado)

    name = f"N{N}B"
    print(f"\n=== Loading {name} ===")
    tpm = cargar_mpt(name)
    print(f"TPM shape: {tpm.shape}")

    # Flujo: tpm + params → subsistema (fuera); estrategia recibe subsistema y retorna Solution
    sol = ejecutar(tpm, config, "phi")
    print(sol)


if __name__ == "__main__":
    main()
