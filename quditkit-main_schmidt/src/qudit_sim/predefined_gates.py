from .gate_class import Gate
from . import gates

# ---------------------------------------------------------------------------
# Predefined gate instances -------------------------------------------------
I_gate     = Gate("I",    1, gates.I,    gates.I_dag)
X_gate     = Gate("X",    1, gates.X,    gates.X_dag)
Y_gate     = Gate("Y",    1, gates.Y,    gates.Y_dag)
Z_gate     = Gate("Z",    1, gates.Z,    gates.Z_dag)
H_gate     = Gate("H",    1, gates.H,    gates.H_dag)
S_gate     = Gate("S",    1, gates.S,    gates.S_dag)
CNOT_gate  = Gate("CNOT", 2, gates.CNOT, gates.CNOT_dag,  gates.apply_CNOT_reshape_idx)
CZ_gate    = Gate("CZ",   2, gates.CZ,   gates.CZ_dag,    gates.apply_CZ_reshape_idx)
SWAP_gate  = Gate("SWAP", 2, gates.SWAP, gates.SWAP_dag,  gates.apply_SWAP_reshape_idx)
W_gate     = Gate("W_{a,b}", 1, None, None, apply_fn=gates.apply_W_reshape_idx, a=None, b=None)
NOISE_gate = Gate("NOISE_CHANNEL",1, None, None, color="#FF0000")       # placeholder gate for noise modeling (default=#FF0000)
M_gate     = Gate("M",    1, None, None)       # measurement
P_gate     = Gate("P",    1, None, None)       # placeholder gate
GROUP_gate = Gate("GROUP:U(φ)", 2, None, None) # vertical box covering multiple qudits,
                                               # visible text is specified after the colon

