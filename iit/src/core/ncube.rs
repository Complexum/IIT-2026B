// ============ NCUBE ============

use std::fmt;

use super::helpers::{bit_at, pow2};

#[derive(Debug, Clone)]
pub struct NCube {
    pub indice: u8,
    pub dims: Vec<u8>,  // ordenado ascendente
    pub data: Vec<f32>, // len = 2^dims.len()
}

impl NCube {
    /// Constructor público con validación (usar desde fronteras)
    pub fn new(indice: u8, mut dims: Vec<u8>, data: Vec<f32>) -> Result<Self, &'static str> {
        dims.sort_unstable();

        // Validar duplicados
        if dims.windows(2).any(|w| w[0] == w[1]) {
            return Err("dims tiene valores duplicados");
        }

        // Validar longitud
        let expected = pow2(dims.len());
        if data.len() != expected {
            return Err("data.len() no coincide con 2^dims.len()");
        }

        Ok(Self { indice, dims, data })
    }

    /// Constructor interno SIN validación (para operaciones internas)
    #[inline]
    fn unchecked(indice: u8, dims: Vec<u8>, data: Vec<f32>) -> Self {
        Self { indice, dims, data }
    }

    #[inline]
    fn pos_of_dim(&self, dim: u8) -> Option<usize> {
        self.dims.iter().position(|&d| d == dim)
    }

    /// Condicionar sin validación - asume inputs correctos
    pub fn condicionar(&self, indices: &[u8], estado_inicial: &[u8]) -> Self {
        // Recolectar posiciones a fijar
        let fixed: Vec<(usize, u8)> = indices
            .iter()
            .filter_map(|&dim| {
                self.pos_of_dim(dim)
                    .map(|pos| (pos, estado_inicial[dim as usize] & 1))
            })
            .collect();

        if fixed.is_empty() {
            return self.clone();
        }

        // Nuevas dims = las que NO están en indices
        let new_dims: Vec<u8> = self
            .dims
            .iter()
            .filter(|d| !indices.contains(d))
            .copied()
            .collect();

        // Filtrar data: solo índices donde todos los bits fijos coinciden
        let new_data: Vec<f32> = (0..self.data.len())
            .filter(|&i| fixed.iter().all(|&(pos, want)| bit_at(i, pos) == want))
            .map(|i| self.data[i])
            .collect();

        Self::unchecked(self.indice, new_dims, new_data)
    }

    /// Marginalizar sin validación
    pub fn marginalizar(&self, ejes: &[u8]) -> Self {
        // Posiciones a marginalizar (ordenadas)
        let mut mpos: Vec<usize> = ejes
            .iter()
            .filter_map(|&dim| self.pos_of_dim(dim))
            .collect();
        mpos.sort_unstable();
        mpos.dedup();

        if mpos.is_empty() {
            return self.clone();
        }

        // Dims restantes
        let new_dims: Vec<u8> = self
            .dims
            .iter()
            .filter(|d| !ejes.contains(d))
            .copied()
            .collect();

        let k = self.dims.len();
        let m = mpos.len();
        let new_k = k - m;

        // Posiciones de bits que se mantienen
        let rpos: Vec<usize> = (0..k).filter(|p| !mpos.contains(p)).collect();

        // Calcular promedios
        let denom = pow2(m) as f32;
        let new_data: Vec<f32> = (0..pow2(new_k))
            .map(|out_idx| {
                // Construir índice base con bits remanentes
                let base_idx: usize = rpos
                    .iter()
                    .enumerate()
                    .map(|(j, &pos)| (bit_at(out_idx, j) as usize) << pos)
                    .fold(0, |acc, x| acc | x);

                // Sumar todas las combinaciones de bits marginalizados
                let sum: f32 = (0..pow2(m))
                    .map(|comb| {
                        let full = mpos.iter().enumerate().fold(base_idx, |acc, (j, &pos)| {
                            acc | ((bit_at(comb, j) as usize) << pos)
                        });
                        self.data[full]
                    })
                    .sum();

                sum / denom
            })
            .collect();

        Self::unchecked(self.indice, new_dims, new_data)
    }

    /// Obtener valor en un estado específico
    #[inline]
    pub fn value_at(&self, estado_inicial: &[u8]) -> f32 {
        if self.dims.is_empty() {
            return self.data[0];
        }
        let idx: usize = self
            .dims
            .iter()
            .enumerate()
            .map(|(pos, &dim)| ((estado_inicial[dim as usize] & 1) as usize) << pos)
            .fold(0, |acc, x| acc | x);
        self.data[idx]
    }
}

// ============ DISPLAY BONITO ============

impl NCube {
    fn fmt_numpy_style(&self, f: &mut fmt::Formatter<'_>, base_indent: usize) -> fmt::Result {
        let depth = self.dims.len();
        if depth == 0 {
            return write!(f, "{:.4}", self.data[0]);
        }

        let num_pairs: usize = 1 << (depth - 1); // 2^(depth-1) pares de valores

        for pair_idx in 0usize..num_pairs {
            // <-- Añade "usize" aquí
            let data_idx = pair_idx * 2;

            // Cuántos brackets abrir: depth si es el primero, sino trailing_zeros + 1
            let opens = if pair_idx == 0 {
                depth
            } else {
                pair_idx.trailing_zeros() as usize + 1
            };

            // Cuántos brackets cerrar: depth si es el último, sino trailing_zeros del siguiente + 1
            let closes = if pair_idx == num_pairs - 1 {
                depth
            } else {
                (pair_idx + 1).trailing_zeros() as usize + 1
            };

            // Nueva línea e indentación (excepto el primero)
            if pair_idx > 0 {
                let indent = base_indent + depth - opens;
                write!(f, "\n{:width$}", "", width = indent)?;
            }

            // Brackets de apertura
            for _ in 0..opens {
                write!(f, "[")?;
            }

            // Datos del par
            write!(
                f,
                "{:.4} {:.4}",
                self.data[data_idx],
                self.data[data_idx + 1]
            )?;

            // Brackets de cierre
            for _ in 0..closes {
                write!(f, "]")?;
            }
        }

        Ok(())
    }
}

impl fmt::Display for NCube {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "NCube(indice={}):", self.indice)?;
        writeln!(f, "  dims={:?}", self.dims)?;
        writeln!(f, "  shape=({})", vec!["2"; self.dims.len()].join(", "))?;
        write!(f, "  data=\n      ")?;
        self.fmt_numpy_style(f, 6)
    }
}
