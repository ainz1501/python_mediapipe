import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modules import triangulate3DHPE as hpe
import math
import numpy

R1 = hpe.R_left
K1 = hpe.K_left
T1 = hpe.T_left

K2 = hpe.K_right
R2 = hpe.R_right
T2 = hpe.T_right

baseline = T2 - T1
baselineNorm = sum([i**2 for i in baseline])
print(baselineNorm)

rotation = R2 - R1 
print(rotation)
