from src.io.manager import listar_redes, cargar_mpt, crear_sistema


def main():
    """CLI entry point for MIP solver."""
    networks = listar_redes()
    print("MIP-IIT Solver")
    print(f"Available networks: {networks}")

    if not networks:
        print("No networks found. Use the TUI to generate one.")
        return

    name = networks[2]
    tpm = cargar_mpt(name)
    estado = tuple(0 for _ in range(tpm.shape[1]))
    sistema = crear_sistema(tpm, estado)

    print(f"Estado inicial {estado}")
    print(f"Loaded {name}: {len(tpm.shape)} states, {tpm.shape[1]} dims")
    print(f"System indices: {sistema.indices}")
    print(f"Marginal distribution: {sistema.distribucion_marginal()}")


if __name__ == "__main__":
    main()
