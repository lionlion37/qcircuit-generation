"""Qudit-Kit based quantum circuit backend."""

# %% auto 0
__all__ = ['get_number_of_gate_params', 'instruction_name_to_qiskit_gate', 'get_target_control_qubits',
           'CircuitsQiskitBackend']

# %% ../../../src/platform/backends/circuits_qiskit.ipynb 2
from ...imports import *
from .base_backend import BaseBackend
from ..circuits_instructions import CircuitInstructions
from ..tokenizer.base_tokenizer import Vocabulary
from ...utils.config_loader import get_obj_from_str

from qudit_sim.circuit import QuantumCircuit #, transpile
from qudit_sim.gate_class import Gate
from qudit_sim.tableau import Tableau
from qudit_sim.utils import *

from qiskit import QuantumCircuit as QiskitCircuit
from qiskit import transpile

# def get_number_of_gate_params(gate_cls: type[Gate]) -> int:
#     # python: gives you the number of any arguments BEFORE *args, minus the ones that have a default, -1 for self parameter of classes
#    return gate_cls.__init__.__code__.co_argcount - len(gate_cls.__init__.__defaults__) - 1

import functools
@functools.lru_cache(None)
def instruction_name_to_quditsim_gate(name: str) -> Gate:
    match name:
        case "cp":
            name = "CPhase"
        case "cx":
            name = "CNOT"
        case _:
            name = name.upper()

    return get_obj_from_str(f"qudit_sim.predefined_gates.{name}_gate")


def get_target_control_qubits(op: tuple) -> Tuple[List[int], List[int]]:
    """Get the target and control qubits of a Quditkit Operation"""
    # TODO: think about if there could be gates with more than one control/target qubit

    acts_on = op[1]

    target_qubit = [acts_on[0]]

    if len(acts_on) == 1:
        control_qubit = []

    else:
        control_qubit = [acts_on[1]]

    return control_qubit, target_qubit


class CircuitsQuditkitBackend(BaseBackend):
    BASIC_BACKEND_TYPE = type[QuantumCircuit]

    def backend_to_genqc(self, qc: QuantumCircuit, ignore_barriers: bool = True) -> CircuitInstructions:
        """Convert a given Quditkit `QuantumCircuit` to genQC `CircuitInstructions`."""
        ops = qc.ops

        instructions = CircuitInstructions(tensor_shape=torch.Size([qc.n, len(ops)]))

        for op in ops:
            control_qubits, target_qubits = get_target_control_qubits(op)

            name = op[0].name.lower()
            if name == "cnot":
                name = "cx"

            instructions.add_instruction(name, control_qubits, target_qubits, [0.0] * 2)

        return instructions

    def genqc_to_backend(self,
                         instructions: CircuitInstructions,
                         place_barriers: bool = True,
                         ignore_errors: bool = False,
                         place_error_placeholders: bool = False) -> QuantumCircuit:
        """Convert given genQC `CircuitInstructions` to a quditkit `QuantumCircuit`."""

        gate_classes = {name: instruction_name_to_quditsim_gate(name) for name in instructions.instruction_names_set}
        qc = QuantumCircuit(num_qudits=instructions.num_qubits, dim=2)  # qudit dimension is fixed to 2 for qubits

        for instruction in instructions.data:
            gate_cls = gate_classes[instruction.name]

            control_qubits, target_qubits = instruction.control_nodes, instruction.target_nodes

            try:
                qc.append(gate_cls, (*control_qubits, *target_qubits), dagger=False)  # TODO: check if dagger gates could be appended
            except Exception as err:
                if ignore_errors:
                    continue
                raise err

        return qc

    def get_unitary(self, qc: QuantumCircuit, remove_global_phase: bool = True) -> np.ndarray:
        raise NotImplementedError()

    def schmidt_rank_vector(self, qc: Optional[QuantumCircuit] = None,
                            tableau: Optional[Tableau] = None) -> List[int]:
        """Return the SRV of a `qudit_sim.QuantumCircuit`."""

        if not exists(tableau):
            tableau = Tableau.zero_state(n=qc.n, d=qc.d)
            tableau.apply_circuit(qc)

        _, rank_vector = schmidt_rank_vector(tableau)

        return list(rank_vector)

    def optimize_circuit(self,
                         qc: QuantumCircuit,
                         vocabulary: Vocabulary,
                         optimization_level: int = 1,
                         silent: bool = True) -> QuantumCircuit:
        """Use `qiskit.compiler.transpile` to optimize a circuit."""


        if optimization_level == 0:
            return qc

        qiskit_circuit = qc.to_qiskit()

        while optimization_level > 0:
            try:
                qc_opt = transpile(qiskit_circuit, optimization_level=optimization_level, basis_gates=vocabulary)
                return QuantumCircuit.from_qiskit(qc_opt)

            except Exception as er:
                if not silent: print(er)
                pass

            optimization_level -= 1

        return QuantumCircuit.from_qiskit(qiskit_circuit)

    def rnd_circuit(self,
                    num_of_qubits: int,
                    num_of_gates: int,
                    gate_pool: Union[Sequence[Gate], Sequence[str]],
                    rng: np.random.Generator) -> QuantumCircuit:
        """Create a random `QuantumCircuit`."""

        qc = QuantumCircuit(num_qudits=num_of_qubits, dim=2)  # qudit dimension is fixed to 2 for qubits
        gate_indices = rng.choice(len(gate_pool), num_of_gates)

        gate_pool = list(gate_pool)
        if isinstance(gate_pool[0], str):
            gate_pool = [instruction_name_to_quditsim_gate(gate) for gate in gate_pool]

        for gate_index in gate_indices:
            gate_cls = gate_pool[gate_index]

            gate = gate_cls

            act_qubits = rng.choice(num_of_qubits, gate.n_qudits,
                                    replace=False)  # order: (*act_qubits)=(*control_qubits, *target_qubits)
            qc.append(gate, tuple(act_qubits), dagger=False)

        return qc

    def randomize_params(self, qc: QuantumCircuit, rng: np.random.Generator) -> QuantumCircuit:
        raise NotImplementedError()

    def draw(self, qc: QuantumCircuit, **kwargs) -> None:
        """Draw the given `QuantumCircuit`."""
        return qc.draw("mpl", **kwargs)
        # plt.show()
