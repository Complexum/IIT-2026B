from src.iit.core.params import Params
from src.iit.strategies.runner import ejecutar
from src.io.manager import cargar_mpt

tpm = cargar_mpt("N7A")
params = Params("1000000", "1111100", "1010111", "0101111")
sol = ejecutar(tpm, params, "phi")
print("IIT4.0!", sol.perdida, sol.particion)
