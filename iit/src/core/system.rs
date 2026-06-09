// ============ SYSTEM ============

use std::fmt;

use super::ncube::NCube;

#[derive(Debug, Clone)]
pub struct System {
    pub estado_inicial: Vec<u8>,
    pub ncubos: Vec<NCube>,
}

impl System {
    pub fn new(estado_inicial: Vec<u8>, ncubos: Vec<NCube>) -> Self {
        Self {
            estado_inicial,
            ncubos,
        }
    }

    pub fn indices_ncubos(&self) -> Vec<u8> {
        self.ncubos.iter().map(|c| c.indice).collect()
    }

    pub fn dims_ncubos(&self) -> Vec<u8> {
        self.ncubos
            .first()
            .map(|c| c.dims.clone())
            .unwrap_or_default()
    }

    pub fn condicionar(&self, indices: &[u8]) -> Self {
        let nuevos: Vec<NCube> = self
            .ncubos
            .iter()
            .filter(|c| !indices.contains(&c.indice))
            .map(|c| c.condicionar(indices, &self.estado_inicial))
            .collect();

        Self::new(self.estado_inicial.clone(), nuevos)
    }

    pub fn substraer(&self, alcance_idx: &[u8], mecanismo_dims: &[u8]) -> Self {
        let nuevos: Vec<NCube> = self
            .ncubos
            .iter()
            .filter(|c| !alcance_idx.contains(&c.indice))
            .map(|c| c.marginalizar(mecanismo_dims))
            .collect();

        Self::new(self.estado_inicial.clone(), nuevos)
    }

    // pub fn bipartir(&self, alcance: &[u8], mecanismo: &[u8]) -> Self {
    //     let nuevos: Vec<NCube> = self
    //         .ncubos
    //         .iter()
    //         .map(|c| {
    //             if alcance.contains(&c.indice) {
    //                 let complement: Vec<u8> = c
    //                     .dims
    //                     .iter()
    //                     .filter(|d| !mecanismo.contains(d))
    //                     .copied()
    //                     .collect();
    //                 c.marginalizar(&complement)
    //             } else {
    //                 c.marginalizar(mecanismo)
    //             }
    //         })
    //         .collect();

    //     Self::new(self.estado_inicial.clone(), nuevos)
    // }

    pub fn distribucion_marginal(&self) -> Vec<f32> {
        self.ncubos
            .iter()
            .map(|c| c.value_at(&self.estado_inicial))
            .collect()
    }
}

impl fmt::Display for System {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "System(indices={:?}, dims={:?})",
            self.indices_ncubos(),
            self.dims_ncubos()
        )?;
        writeln!(f, "  estado_inicial={:?}", self.estado_inicial)?;
        writeln!(f, "  NCubes:")?;
        for c in &self.ncubos {
            writeln!(f, "    {}", c)?;
        }
        Ok(())
    }
}
 