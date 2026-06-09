mod core;

use core::ncube::NCube;
use core::system::System;

// ============ MAIN ============

fn main() {
    let estado = vec![1, 0, 0];

    // Crear cubos manualmente para el ejemplo
    let c0 = NCube::new(
        0,
        vec![0, 1, 2],
        vec![0.5, 0.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    )
    .unwrap();
    let c1 = NCube::new(
        1,
        vec![0, 1, 2],
        vec![0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0],
    )
    .unwrap();
    let c2 = NCube::new(
        2,
        vec![0, 1, 2],
        vec![0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0],
    )
    .unwrap();

    let sys = System::new(estado.clone(), vec![c0, c1, c2]);
    println!("=== Sistema Original ===");
    println!("{}", sys);

    // let condicionado = sys.condicionar(&[2]);
    // println!("=== Después de condicionar en [2] ===");
    // println!("{}", condicionado);

    let substraido = sys.substraer(&[0], &[2]);
    println!("=== Después de substraer alcance=[0], mecanismo=[2] ===");
    println!("{}", substraido);

    let dist = substraido.distribucion_marginal();
    println!("Distribución marginal: {:?}", dist);
}
