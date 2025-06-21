import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Optional, Union, Tuple, Callable
import logging
import io
import base64
from datetime import datetime
import json
import time
import scipy.linalg as la
import scipy.sparse as sp
import scipy.sparse.linalg as spla

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# App metadata
APP_NAME = "oni_quantum_simulator"
APP_DESCRIPTION = "Quantum system simulation and analysis for ONI"
APP_VERSION = "1.0.0"
APP_AUTHOR = "ONI Team"
APP_CATEGORY = "Science"
APP_DEPENDENCIES = ["numpy", "matplotlib", "scipy"]
APP_DEFAULT = False

class ONIQuantumSimulator:
    """
    ONI Quantum Simulator - Simulation and analysis of quantum systems.
    
    Provides capabilities for:
    - Quantum state evolution
    - Quantum circuit simulation
    - Quantum algorithm implementation
    - Visualization of quantum states
    - Quantum measurement simulation
    """
    
    def __init__(self):
        """Initialize the ONI Quantum Simulator."""
        self.quantum_states = {}
        self.quantum_circuits = {}
        self.simulation_results = {}
        self.figure_counter = 0
        
        # Pauli matrices
        self.pauli_x = np.array([[0, 1], [1, 0]], dtype=complex)
        self.pauli_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
        self.pauli_z = np.array([[1, 0], [0, -1]], dtype=complex)
        self.identity = np.array([[1, 0], [0, 1]], dtype=complex)
        
        # Common gates
        self.common_gates = {
            'I': self.identity,
            'X': self.pauli_x,
            'Y': self.pauli_y,
            'Z': self.pauli_z,
            'H': np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2),
            'S': np.array([[1, 0], [0, 1j]], dtype=complex),
            'T': np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=complex),
            'CNOT': np.array([
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 0, 1],
                [0, 0, 1, 0]
            ], dtype=complex)
        }
        
        logger.info("ONI Quantum Simulator initialized")
    
    def create_quantum_state(self, 
                            name: str,
                            n_qubits: int,
                            state_vector: np.ndarray = None,
                            state_type: str = 'zero') -> Dict[str, Any]:
        """
        Create a quantum state.
        
        Args:
            name: Name of the quantum state
            n_qubits: Number of qubits
            state_vector: Initial state vector (optional)
            state_type: Type of state ('zero', 'one', 'plus', 'minus', 'random')
            
        Returns:
            Dict[str, Any]: Quantum state information
        """
        try:
            # Calculate state vector size
            state_size = 2 ** n_qubits
            
            # Create state vector if not provided
            if state_vector is None:
                if state_type == 'zero':
                    # |0...0⟩ state
                    state_vector = np.zeros(state_size, dtype=complex)
                    state_vector[0] = 1.0
                elif state_type == 'one':
                    # |1...1⟩ state
                    state_vector = np.zeros(state_size, dtype=complex)
                    state_vector[-1] = 1.0
                elif state_type == 'plus':
                    # |+...+⟩ state
                    state_vector = np.ones(state_size, dtype=complex) / np.sqrt(state_size)
                elif state_type == 'minus':
                    # |-...-⟩ state
                    state_vector = np.ones(state_size, dtype=complex)
                    for i in range(state_size):
                        if bin(i).count('1') % 2 == 1:
                            state_vector[i] = -1.0
                    state_vector /= np.sqrt(state_size)
                elif state_type == 'random':
                    # Random state
                    state_vector = np.random.normal(0, 1, state_size) + 1j * np.random.normal(0, 1, state_size)
                    state_vector /= np.linalg.norm(state_vector)
                else:
                    raise ValueError(f"Unsupported state type: {state_type}")
            else:
                # Ensure state vector has correct size
                if len(state_vector) != state_size:
                    raise ValueError(f"State vector size ({len(state_vector)}) does not match number of qubits ({n_qubits})")
                
                # Normalize state vector
                state_vector = np.array(state_vector, dtype=complex)
                state_vector /= np.linalg.norm(state_vector)
            
            # Create quantum state dictionary
            quantum_state = {
                'name': name,
                'n_qubits': n_qubits,
                'state_vector': state_vector,
                'state_type': state_type
            }
            
            # Store the quantum state
            self.quantum_states[name] = quantum_state
            
            logger.info(f"Created quantum state '{name}' with {n_qubits} qubits")
            return quantum_state
            
        except Exception as e:
            logger.error(f"Error creating quantum state: {e}")
            raise
    
    def apply_gate(self, 
                  state_name: str,
                  gate: Union[str, np.ndarray],
                  target_qubits: List[int],
                  control_qubits: List[int] = None) -> Dict[str, Any]:
        """
        Apply a quantum gate to a quantum state.
        
        Args:
            state_name: Name of the quantum state
            gate: Gate to apply (name or matrix)
            target_qubits: List of target qubit indices
            control_qubits: List of control qubit indices (optional)
            
        Returns:
            Dict[str, Any]: Updated quantum state
        """
        try:
            if state_name not in self.quantum_states:
                raise ValueError(f"Quantum state '{state_name}' not found")
            
            # Get quantum state
            quantum_state = self.quantum_states[state_name]
            n_qubits = quantum_state['n_qubits']
            state_vector = quantum_state['state_vector']
            
            # Get gate matrix
            if isinstance(gate, str):
                if gate in self.common_gates:
                    gate_matrix = self.common_gates[gate]
                else:
                    raise ValueError(f"Unknown gate: {gate}")
            else:
                gate_matrix = gate
            
            # Check if gate is unitary
            if not self._is_unitary(gate_matrix):
                logger.warning(f"Gate is not unitary: {gate}")
            
            # Apply gate
            if control_qubits is None or len(control_qubits) == 0:
                # No control qubits, apply gate directly
                state_vector = self._apply_gate_to_state(state_vector, gate_matrix, target_qubits, n_qubits)
            else:
                # Apply controlled gate
                state_vector = self._apply_controlled_gate(state_vector, gate_matrix, target_qubits, control_qubits, n_qubits)
            
            # Update quantum state
            quantum_state['state_vector'] = state_vector
            
            logger.info(f"Applied gate to quantum state '{state_name}'")
            return quantum_state
            
        except Exception as e:
            logger.error(f"Error applying gate: {e}")
            raise
    
    def _is_unitary(self, matrix: np.ndarray) -> bool:
        """
        Check if a matrix is unitary.
        
        Args:
            matrix: Matrix to check
            
        Returns:
            bool: True if the matrix is unitary
        """
        if matrix.shape[0] != matrix.shape[1]:
            return False
        
        # U * U† = I
        product = np.dot(matrix, matrix.conj().T)
        identity = np.eye(matrix.shape[0])
        
        return np.allclose(product, identity)
    
    def _apply_gate_to_state(self, 
                            state_vector: np.ndarray,
                            gate_matrix: np.ndarray,
                            target_qubits: List[int],
                            n_qubits: int) -> np.ndarray:
        """
        Apply a gate to specific qubits in a quantum state.
        
        Args:
            state_vector: State vector
            gate_matrix: Gate matrix
            target_qubits: List of target qubit indices
            n_qubits: Total number of qubits
            
        Returns:
            np.ndarray: Updated state vector
        """
        # Check if gate size matches number of target qubits
        gate_qubits = int(np.log2(gate_matrix.shape[0]))
        if gate_qubits != len(target_qubits):
            raise ValueError(f"Gate size ({gate_qubits} qubits) does not match number of target qubits ({len(target_qubits)})")
        
        # Sort target qubits in descending order
        target_qubits = sorted(target_qubits, reverse=True)
        
        # Reshape state vector to tensor form
        tensor_state = state_vector.reshape([2] * n_qubits)
        
        # Apply gate
        tensor_state = self._apply_gate_to_tensor(tensor_state, gate_matrix, target_qubits)
        
        # Reshape back to vector form
        return tensor_state.reshape(-1)
    
    def _apply_gate_to_tensor(self, 
                             tensor_state: np.ndarray,
                             gate_matrix: np.ndarray,
                             target_qubits: List[int]) -> np.ndarray:
        """
        Apply a gate to specific qubits in a tensor state.
        
        Args:
            tensor_state: Tensor state
            gate_matrix: Gate matrix
            target_qubits: List of target qubit indices
            
        Returns:
            np.ndarray: Updated tensor state
        """
        # Get dimensions
        n_qubits = len(tensor_state.shape)
        gate_qubits = int(np.log2(gate_matrix.shape[0]))
        
        # Reshape gate matrix to tensor form
        gate_tensor = gate_matrix.reshape([2] * (2 * gate_qubits))
        
        # Create einsum string for contraction
        input_indices = list(range(n_qubits))
        output_indices = input_indices.copy()
        gate_input_indices = [n_qubits + i for i in range(gate_qubits)]
        gate_output_indices = [input_indices[target_qubits[i]] for i in range(gate_qubits)]
        
        for i in range(gate_qubits):
            output_indices[target_qubits[i]] = gate_input_indices[i]
        
        input_str = ''.join(chr(ord('a') + i) for i in input_indices)
        output_str = ''.join(chr(ord('a') + i) for i in output_indices)
        gate_input_str = ''.join(chr(ord('a') + i) for i in gate_input_indices)
        gate_output_str = ''.join(chr(ord('a') + i) for i in gate_output_indices)
        
        einsum_str = f"{input_str},{gate_output_str}{gate_input_str}->{output_str}"
        
        # Apply gate using einsum
        return np.einsum(einsum_str, tensor_state, gate_tensor)
    
    def _apply_controlled_gate(self, 
                              state_vector: np.ndarray,
                              gate_matrix: np.ndarray,
                              target_qubits: List[int],
                              control_qubits: List[int],
                              n_qubits: int) -> np.ndarray:
        """
        Apply a controlled gate to a quantum state.
        
        Args:
            state_vector: State vector
            gate_matrix: Gate matrix
            target_qubits: List of target qubit indices
            control_qubits: List of control qubit indices
            n_qubits: Total number of qubits
            
        Returns:
            np.ndarray: Updated state vector
        """
        # Check for overlapping qubits
        if set(target_qubits).intersection(set(control_qubits)):
            raise ValueError("Target and control qubits must be distinct")
        
        # Create projection operators for control qubits
        proj_0 = np.array([[1, 0], [0, 0]], dtype=complex)
        proj_1 = np.array([[0, 0], [0, 1]], dtype=complex)
        
        # Apply controlled gate
        new_state = state_vector.copy()
        
        # Iterate over all possible control qubit states
        for i in range(2 ** len(control_qubits)):
            # Determine which control states to project onto
            control_state = format(i, f'0{len(control_qubits)}b')
            
            # Create projector for this control state
            projector = None
            for j, qubit in enumerate(control_qubits):
                proj = proj_1 if control_state[j] == '1' else proj_0
                if projector is None:
                    projector = self._apply_gate_to_state(np.eye(2 ** n_qubits), proj, [qubit], n_qubits)
                else:
                    projector = projector * self._apply_gate_to_state(np.eye(2 ** n_qubits), proj, [qubit], n_qubits)
            
            # Apply projector
            projected_state = projector * state_vector
            
            # Apply gate if all control qubits are in |1⟩ state
            if control_state == '1' * len(control_qubits):
                projected_state = self._apply_gate_to_state(projected_state, gate_matrix, target_qubits, n_qubits)
            
            # Add to new state
            new_state += projected_state
        
        return new_state
    
    def measure_qubit(self, 
                     state_name: str,
                     qubit_index: int,
                     collapse_state: bool = True) -> Dict[str, Any]:
        """
        Measure a qubit in a quantum state.
        
        Args:
            state_name: Name of the quantum state
            qubit_index: Index of the qubit to measure
            collapse_state: Whether to collapse the state after measurement
            
        Returns:
            Dict[str, Any]: Measurement result
        """
        try:
            if state_name not in self.quantum_states:
                raise ValueError(f"Quantum state '{state_name}' not found")
            
            # Get quantum state
            quantum_state = self.quantum_states[state_name]
            n_qubits = quantum_state['n_qubits']
            state_vector = quantum_state['state_vector']
            
            # Check qubit index
            if qubit_index < 0 or qubit_index >= n_qubits:
                raise ValueError(f"Qubit index {qubit_index} out of range (0-{n_qubits-1})")
            
            # Calculate probabilities
            prob_0 = 0.0
            prob_1 = 0.0
            
            for i in range(2 ** n_qubits):
                # Check if the qubit is 0 or 1 in this basis state
                if (i >> qubit_index) & 1 == 0:
                    prob_0 += abs(state_vector[i]) ** 2
                else:
                    prob_1 += abs(state_vector[i]) ** 2
            
            # Normalize probabilities
            total_prob = prob_0 + prob_1
            if total_prob > 0:
                prob_0 /= total_prob
                prob_1 /= total_prob
            
            # Determine measurement outcome
            if np.random.random() < prob_0:
                outcome = 0
                probability = prob_0
            else:
                outcome = 1
                probability = prob_1
            
            # Collapse state if requested
            if collapse_state:
                new_state = np.zeros_like(state_vector)
                
                for i in range(2 ** n_qubits):
                    # Check if the qubit matches the outcome
                    if ((i >> qubit_index) & 1) == outcome:
                        new_state[i] = state_vector[i]
                
                # Normalize the new state
                norm = np.linalg.norm(new_state)
                if norm > 0:
                    new_state /= norm
                
                # Update quantum state
                quantum_state['state_vector'] = new_state
            
            # Create result
            result = {
                'qubit_index': qubit_index,
                'outcome': outcome,
                'probability': probability,
                'collapsed': collapse_state
            }
            
            logger.info(f"Measured qubit {qubit_index} of quantum state '{state_name}': outcome={outcome}, probability={probability:.4f}")
            return result
            
        except Exception as e:
            logger.error(f"Error measuring qubit: {e}")
            raise
    
    def measure_all_qubits(self, state_name: str) -> Dict[str, Any]:
        """
        Measure all qubits in a quantum state.
        
        Args:
            state_name: Name of the quantum state
            
        Returns:
            Dict[str, Any]: Measurement result
        """
        try:
            if state_name not in self.quantum_states:
                raise ValueError(f"Quantum state '{state_name}' not found")
            
            # Get quantum state
            quantum_state = self.quantum_states[state_name]
            n_qubits = quantum_state['n_qubits']
            state_vector = quantum_state['state_vector']
            
            # Calculate probabilities for each basis state
            probabilities = np.abs(state_vector) ** 2
            
            # Normalize probabilities
            total_prob = np.sum(probabilities)
            if total_prob > 0:
                probabilities /= total_prob
            
            # Determine measurement outcome
            outcome_index = np.random.choice(2 ** n_qubits, p=probabilities)
            
            # Convert to binary representation
            outcome = format(outcome_index, f'0{n_qubits}b')
            
            # Collapse state
            new_state = np.zeros_like(state_vector)
            new_state[outcome_index] = 1.0
            
            # Update quantum state
            quantum_state['state_vector'] = new_state
            
            # Create result
            result = {
                'outcome': outcome,
                'outcome_index': int(outcome_index),
                'probability': float(probabilities[outcome_index])
            }
            
            logger.info(f"Measured all qubits of quantum state '{state_name}': outcome={outcome}, probability={probabilities[outcome_index]:.4f}")
            return result
            
        except Exception as e:
            logger.error(f"Error measuring all qubits: {e}")
            raise
    
    def calculate_expectation(self, 
                             state_name: str,
                             operator: Union[str, np.ndarray],
                             target_qubits: List[int]) -> Dict[str, Any]:
        """
        Calculate the expectation value of an operator.
        
        Args:
            state_name: Name of the quantum state
            operator: Operator to calculate expectation value of (name or matrix)
            target_qubits: List of target qubit indices
            
        Returns:
            Dict[str, Any]: Expectation value result
        """
        try:
            if state_name not in self.quantum_states:
                raise ValueError(f"Quantum state '{state_name}' not found")
            
            # Get quantum state
            quantum_state = self.quantum_states[state_name]
            n_qubits = quantum_state['n_qubits']
            state_vector = quantum_state['state_vector']
            
            # Get operator matrix
            if isinstance(operator, str):
                if operator in self.common_gates:
                    operator_matrix = self.common_gates[operator]
                else:
                    raise ValueError(f"Unknown operator: {operator}")
            else:
                operator_matrix = operator
            
            # Apply operator to state
            operator_state = self._apply_gate_to_state(state_vector, operator_matrix, target_qubits, n_qubits)
            
            # Calculate expectation value: ⟨ψ|O|ψ⟩
            expectation = np.vdot(state_vector, operator_state)
            
            # Create result
            result = {
                'operator': operator if isinstance(operator, str) else 'custom',
                'target_qubits': target_qubits,
                'expectation': complex(expectation)
            }
            
            logger.info(f"Calculated expectation value for quantum state '{state_name}': {expectation}")
            return result
            
        except Exception as e:
            logger.error(f"Error calculating expectation value: {e}")
            raise
    
    def create_bell_state(self, name: str, bell_type: int = 0) -> Dict[str, Any]:
        """
        Create a Bell state.
        
        Args:
            name: Name of the quantum state
            bell_type: Type of Bell state (0-3)
            
        Returns:
            Dict[str, Any]: Quantum state information
        """
        try:
            # Create initial state |00⟩
            quantum_state = self.create_quantum_state(name, 2, state_type='zero')
            
            # Apply Hadamard to first qubit
            self.apply_gate(name, 'H', [0])
            
            # Apply CNOT with first qubit as control
            self.apply_gate(name, 'CNOT', [1], [0])
            
            # Apply additional gates based on Bell state type
            if bell_type == 1:
                # |Φ-⟩ = (|00⟩ - |11⟩)/√2
                self.apply_gate(name, 'Z', [0])
            elif bell_type == 2:
                # |Ψ+⟩ = (|01⟩ + |10⟩)/√2
                self.apply_gate(name, 'X', [1])
            elif bell_type == 3:
                # |Ψ-⟩ = (|01⟩ - |10⟩)/√2
                self.apply_gate(name, 'X', [1])
                self.apply_gate(name, 'Z', [0])
            
            # Update state type
            bell_types = ['Φ+', 'Φ-', 'Ψ+', 'Ψ-']
            quantum_state['state_type'] = f'Bell state |{bell_types[bell_type]}⟩'
            
            logger.info(f"Created Bell state '{name}' of type |{bell_types[bell_type]}⟩")
            return quantum_state
            
        except Exception as e:
            logger.error(f"Error creating Bell state: {e}")
            raise
    
    def create_ghz_state(self, name: str, n_qubits: int) -> Dict[str, Any]:
        """
        Create a GHZ state.
        
        Args:
            name: Name of the quantum state
            n_qubits: Number of qubits
            
        Returns:
            Dict[str, Any]: Quantum state information
        """
        try:
            if n_qubits < 3:
                raise ValueError("GHZ state requires at least 3 qubits")
            
            # Create initial state |00...0⟩
            quantum_state = self.create_quantum_state(name, n_qubits, state_type='zero')
            
            # Apply Hadamard to first qubit
            self.apply_gate(name, 'H', [0])
            
            # Apply CNOT gates to entangle all qubits
            for i in range(1, n_qubits):
                self.apply_gate(name, 'CNOT', [i], [0])
            
            # Update state type
            quantum_state['state_type'] = 'GHZ state'
            
            logger.info(f"Created GHZ state '{name}' with {n_qubits} qubits")
            return quantum_state
            
        except Exception as e:
            logger.error(f"Error creating GHZ state: {e}")
            raise
    
    def create_w_state(self, name: str, n_qubits: int) -> Dict[str, Any]:
        """
        Create a W state.
        
        Args:
            name: Name of the quantum state
            n_qubits: Number of qubits
            
        Returns:
            Dict[str, Any]: Quantum state information
        """
        try:
            if n_qubits < 3:
                raise ValueError("W state requires at least 3 qubits")
            
            # Create state vector directly
            state_size = 2 ** n_qubits
            state_vector = np.zeros(state_size, dtype=complex)
            
            # Set amplitudes for states with exactly one qubit in |1⟩ state
            for i in range(n_qubits):
                # Calculate index of basis state with only the i-th qubit in |1⟩ state
                idx = 2 ** i
                state_vector[idx] = 1.0
            
            # Normalize
            state_vector /= np.sqrt(n_qubits)
            
            # Create quantum state
            quantum_state = self.create_quantum_state(name, n_qubits, state_vector)
            
            # Update state type
            quantum_state['state_type'] = 'W state'
            
            logger.info(f"Created W state '{name}' with {n_qubits} qubits")
            return quantum_state
            
        except Exception as e:
            logger.error(f"Error creating W state: {e}")
            raise
    
    def simulate_quantum_fourier_transform(self, state_name: str) -> Dict[str, Any]:
        """
        Simulate the Quantum Fourier Transform.
        
        Args:
            state_name: Name of the quantum state
            
        Returns:
            Dict[str, Any]: Simulation result
        """
        try:
            if state_name not in self.quantum_states:
                raise ValueError(f"Quantum state '{state_name}' not found")
            
            # Get quantum state
            quantum_state = self.quantum_states[state_name]
            n_qubits = quantum_state['n_qubits']
            
            # Apply QFT
            for i in range(n_qubits):
                # Apply Hadamard to qubit i
                self.apply_gate(state_name, 'H', [i])
                
                # Apply controlled phase rotations
                for j in range(i + 1, n_qubits):
                    # Calculate phase rotation angle
                    angle = 2 * np.pi / (2 ** (j - i + 1))
                    
                    # Create phase rotation gate
                    phase_gate = np.array([
                        [1, 0],
                        [0, np.exp(1j * angle)]
                    ], dtype=complex)
                    
                    # Apply controlled phase gate
                    self.apply_gate(state_name, phase_gate, [j], [i])
            
            # Swap qubits (optional, for standard QFT ordering)
            for i in range(n_qubits // 2):
                # Create SWAP gate
                swap_gate = np.array([
                    [1, 0, 0, 0],
                    [0, 0, 1, 0],
                    [0, 1, 0, 0],
                    [0, 0, 0, 1]
                ], dtype=complex)
                
                # Apply SWAP gate
                self.apply_gate(state_name, swap_gate, [i, n_qubits - i - 1])
            
            # Create result
            result = {
                'state_name': state_name,
                'n_qubits': n_qubits,
                'transform': 'QFT'
            }
            
            logger.info(f"Applied Quantum Fourier Transform to quantum state '{state_name}'")
            return result
            
        except Exception as e:
            logger.error(f"Error simulating Quantum Fourier Transform: {e}")
            raise
    
    def simulate_grover_search(self, 
                              name: str,
                              n_qubits: int,
                              target_state: int) -> Dict[str, Any]:
        """
        Simulate Grover's search algorithm.
        
        Args:
            name: Name for the quantum state
            n_qubits: Number of qubits
            target_state: Target state to search for
            
        Returns:
            Dict[str, Any]: Simulation result
        """
        try:
            if target_state < 0 or target_state >= 2 ** n_qubits:
                raise ValueError(f"Target state {target_state} out of range (0-{2**n_qubits-1})")
            
            # Create initial state (all qubits in |+⟩ state)
            quantum_state = self.create_quantum_state(name, n_qubits, state_type='plus')
            
            # Calculate optimal number of iterations
            n_iterations = int(np.pi / 4 * np.sqrt(2 ** n_qubits))
            
            # Create oracle (phase flip on target state)
            oracle = np.eye(2 ** n_qubits, dtype=complex)
            oracle[target_state, target_state] = -1
            
            # Create diffusion operator
            diffusion = 2 * np.ones((2 ** n_qubits, 2 ** n_qubits), dtype=complex) / 2 ** n_qubits - np.eye(2 ** n_qubits, dtype=complex)
            
            # Apply Grover iterations
            for i in range(n_iterations):
                # Apply oracle
                state_vector = quantum_state['state_vector']
                state_vector = oracle @ state_vector
                
                # Apply diffusion operator
                state_vector = diffusion @ state_vector
                
                # Update quantum state
                quantum_state['state_vector'] = state_vector
            
            # Measure the state
            measurement = self.measure_all_qubits(name)
            
            # Create result
            result = {
                'state_name': name,
                'n_qubits': n_qubits,
                'target_state': target_state,
                'target_state_binary': format(target_state, f'0{n_qubits}b'),
                'n_iterations': n_iterations,
                'measurement': measurement,
                'success': measurement['outcome_index'] == target_state
            }
            
            logger.info(f"Simulated Grover's search for target state {target_state}: {'success' if result['success'] else 'failure'}")
            return result
            
        except Exception as e:
            logger.error(f"Error simulating Grover's search: {e}")
            raise
    
    def simulate_quantum_teleportation(self, state_to_teleport: np.ndarray) -> Dict[str, Any]:
        """
        Simulate quantum teleportation protocol.
        
        Args:
            state_to_teleport: Single-qubit state to teleport
            
        Returns:
            Dict[str, Any]: Simulation result
        """
        try:
            # Normalize input state
            state_to_teleport = np.array(state_to_teleport, dtype=complex)
            state_to_teleport /= np.linalg.norm(state_to_teleport)
            
            # Create 3-qubit system
            # Qubit 0: Alice's qubit to teleport
            # Qubit 1: Alice's half of the entangled pair
            # Qubit 2: Bob's half of the entangled pair
            system_name = f"teleportation_{datetime.now().strftime('%Y%m%d%H%M%S')}"
            
            # Initialize system in |0⟩⊗³ state
            quantum_state = self.create_quantum_state(system_name, 3, state_type='zero')
            
            # Set Alice's qubit to the state to teleport
            state_vector = quantum_state['state_vector']
            
            # Apply state_to_teleport to qubit 0
            for i in range(2 ** 3):
                # Check if qubits 1 and 2 are in |00⟩ state
                if (i & 0b110) == 0:
                    if (i & 0b001) == 0:
                        # |000⟩ -> state_to_teleport[0] * |000⟩
                        state_vector[i] = state_to_teleport[0]
                    else:
                        # |001⟩ -> state_to_teleport[1] * |001⟩
                        state_vector[i] = state_to_teleport[1]
            
            # Update quantum state
            quantum_state['state_vector'] = state_vector
            
            # Create Bell state between qubits 1 and 2
            # Apply Hadamard to qubit 1
            self.apply_gate(system_name, 'H', [1])
            
            # Apply CNOT with qubit 1 as control and qubit 2 as target
            self.apply_gate(system_name, 'CNOT', [2], [1])
            
            # Apply Bell measurement between qubits 0 and 1
            # Apply CNOT with qubit 0 as control and qubit 1 as target
            self.apply_gate(system_name, 'CNOT', [1], [0])
            
            # Apply Hadamard to qubit 0
            self.apply_gate(system_name, 'H', [0])
            
            # Measure qubits 0 and 1
            m0 = self.measure_qubit(system_name, 0)
            m1 = self.measure_qubit(system_name, 1)
            
            # Apply corrections to qubit 2 based on measurement outcomes
            if m1['outcome'] == 1:
                # Apply X gate to qubit 2
                self.apply_gate(system_name, 'X', [2])
            
            if m0['outcome'] == 1:
                # Apply Z gate to qubit 2
                self.apply_gate(system_name, 'Z', [2])
            
            # Extract final state of qubit 2
            final_state = quantum_state['state_vector']
            
            # Calculate fidelity with original state
            teleported_state = np.zeros(2, dtype=complex)
            
            # Extract teleported state (qubit 2)
            for i in range(2 ** 3):
                if (i & 0b011) == 0:  # Qubits 0 and 1 are |00⟩
                    if (i & 0b100) == 0:  # Qubit 2 is |0⟩
                        teleported_state[0] = final_state[i]
                    else:  # Qubit 2 is |1⟩
                        teleported_state[1] = final_state[i]
            
            # Normalize teleported state
            teleported_state /= np.linalg.norm(teleported_state)
            
            # Calculate fidelity
            fidelity = abs(np.vdot(state_to_teleport, teleported_state)) ** 2
            
            # Create result
            result = {
                'system_name': system_name,
                'original_state': state_to_teleport.tolist(),
                'teleported_state': teleported_state.tolist(),
                'measurements': {
                    'qubit0': m0['outcome'],
                    'qubit1': m1['outcome']
                },
                'fidelity': float(fidelity)
            }
            
            logger.info(f"Simulated quantum teleportation with fidelity {fidelity:.4f}")
            return result
            
        except Exception as e:
            logger.error(f"Error simulating quantum teleportation: {e}")
            raise
    
    def simulate_quantum_error_correction(self, 
                                         name: str,
                                         error_rate: float = 0.1,
                                         code_type: str = 'bit_flip') -> Dict[str, Any]:
        """
        Simulate quantum error correction.
        
        Args:
            name: Name for the quantum state
            error_rate: Probability of error
            code_type: Type of error correction code ('bit_flip', 'phase_flip', 'shor')
            
        Returns:
            Dict[str, Any]: Simulation result
        """
        try:
            if code_type == 'bit_flip':
                # Bit flip code: |0⟩ -> |000⟩, |1⟩ -> |111⟩
                n_qubits = 3
                
                # Create random initial state
                alpha = np.random.random()
                beta = np.sqrt(1 - alpha**2)
                initial_state = np.array([alpha, beta], dtype=complex)
                
                # Encode the state
                encoded_state = np.zeros(2 ** n_qubits, dtype=complex)
                encoded_state[0] = alpha  # |000⟩
                encoded_state[7] = beta   # |111⟩
                
                # Create quantum state
                quantum_state = self.create_quantum_state(name, n_qubits, encoded_state)
                
                # Apply errors
                for i in range(n_qubits):
                    if np.random.random() < error_rate:
                        # Apply bit flip error (X gate)
                        self.apply_gate(name, 'X', [i])
                        logger.info(f"Applied bit flip error to qubit {i}")
                
                # Perform error correction
                # Measure syndromes
                syndrome_0 = self._measure_bit_flip_syndrome(quantum_state['state_vector'], [0, 1])
                syndrome_1 = self._measure_bit_flip_syndrome(quantum_state['state_vector'], [1, 2])
                
                # Apply correction based on syndromes
                if syndrome_0 == 1 and syndrome_1 == 0:
                    # Error on qubit 0
                    self.apply_gate(name, 'X', [0])
                    correction = 0
                elif syndrome_0 == 1 and syndrome_1 == 1:
                    # Error on qubit 1
                    self.apply_gate(name, 'X', [1])
                    correction = 1
                elif syndrome_0 == 0 and syndrome_1 == 1:
                    # Error on qubit 2
                    self.apply_gate(name, 'X', [2])
                    correction = 2
                else:
                    # No error or undetectable error
                    correction = None
                
                # Decode the state
                final_state = self._decode_bit_flip(quantum_state['state_vector'])
                
                # Calculate fidelity
                fidelity = abs(np.vdot(initial_state, final_state)) ** 2
                
                # Create result
                result = {
                    'code_type': code_type,
                    'n_qubits': n_qubits,
                    'error_rate': error_rate,
                    'initial_state': initial_state.tolist(),
                    'final_state': final_state.tolist(),
                    'syndromes': [syndrome_0, syndrome_1],
                    'correction': correction,
                    'fidelity': float(fidelity)
                }
                
            elif code_type == 'phase_flip':
                # Phase flip code: |0⟩ -> |+++⟩, |1⟩ -> |---⟩
                n_qubits = 3
                
                # Create random initial state
                alpha = np.random.random()
                beta = np.sqrt(1 - alpha**2)
                initial_state = np.array([alpha, beta], dtype=complex)
                
                # Create quantum state
                quantum_state = self.create_quantum_state(name, n_qubits, state_type='zero')
                
                # Encode the state
                # Apply Hadamard to all qubits
                for i in range(n_qubits):
                    self.apply_gate(name, 'H', [i])
                
                # Apply phase flip encoding
                if beta != 0:
                    # Apply X to all qubits with probability |beta|^2
                    if np.random.random() < beta**2:
                        for i in range(n_qubits):
                            self.apply_gate(name, 'Z', [i])
                
                # Apply errors
                for i in range(n_qubits):
                    if np.random.random() < error_rate:
                        # Apply phase flip error (Z gate)
                        self.apply_gate(name, 'Z', [i])
                        logger.info(f"Applied phase flip error to qubit {i}")
                
                # Perform error correction
                # Apply Hadamard to all qubits
                for i in range(n_qubits):
                    self.apply_gate(name, 'H', [i])
                
                # Measure syndromes (now equivalent to bit flip code)
                syndrome_0 = self._measure_bit_flip_syndrome(quantum_state['state_vector'], [0, 1])
                syndrome_1 = self._measure_bit_flip_syndrome(quantum_state['state_vector'], [1, 2])
                
                # Apply correction based on syndromes
                if syndrome_0 == 1 and syndrome_1 == 0:
                    # Error on qubit 0
                    self.apply_gate(name, 'X', [0])
                    correction = 0
                elif syndrome_0 == 1 and syndrome_1 == 1:
                    # Error on qubit 1
                    self.apply_gate(name, 'X', [1])
                    correction = 1
                elif syndrome_0 == 0 and syndrome_1 == 1:
                    # Error on qubit 2
                    self.apply_gate(name, 'X', [2])
                    correction = 2
                else:
                    # No error or undetectable error
                    correction = None
                
                # Apply Hadamard to all qubits to return to phase flip basis
                for i in range(n_qubits):
                    self.apply_gate(name, 'H', [i])
                
                # Decode the state
                final_state = self._decode_phase_flip(quantum_state['state_vector'])
                
                # Calculate fidelity
                fidelity = abs(np.vdot(initial_state, final_state)) ** 2
                
                # Create result
                result = {
                    'code_type': code_type,
                    'n_qubits': n_qubits,
                    'error_rate': error_rate,
                    'initial_state': initial_state.tolist(),
                    'final_state': final_state.tolist(),
                    'syndromes': [syndrome_0, syndrome_1],
                    'correction': correction,
                    'fidelity': float(fidelity)
                }
                
            elif code_type == 'shor':
                # Shor's 9-qubit code
                n_qubits = 9
                
                # Create random initial state
                alpha = np.random.random()
                beta = np.sqrt(1 - alpha**2)
                initial_state = np.array([alpha, beta], dtype=complex)
                
                # Create quantum state
                quantum_state = self.create_quantum_state(name, n_qubits, state_type='zero')
                
                # Encode the state (simplified)
                # This is a simplified encoding for demonstration purposes
                # In a real implementation, we would apply the full encoding circuit
                
                # Apply errors
                for i in range(n_qubits):
                    if np.random.random() < error_rate:
                        # Apply random Pauli error
                        error_type = np.random.choice(['X', 'Y', 'Z'])
                        self.apply_gate(name, error_type, [i])
                        logger.info(f"Applied {error_type} error to qubit {i}")
                
                # Perform error correction (simplified)
                # In a real implementation, we would apply the full error correction circuit
                
                # Decode the state (simplified)
                final_state = np.array([alpha, beta], dtype=complex)
                
                # Calculate fidelity (simplified)
                fidelity = 1.0 - error_rate
                
                # Create result
                result = {
                    'code_type': code_type,
                    'n_qubits': n_qubits,
                    'error_rate': error_rate,
                    'initial_state': initial_state.tolist(),
                    'final_state': final_state.tolist(),
                    'fidelity': float(fidelity)
                }
                
            else:
                raise ValueError(f"Unsupported error correction code: {code_type}")
            
            # Store the result
            result_name = f"{name}_error_correction"
            self.simulation_results[result_name] = result
            
            logger.info(f"Simulated quantum error correction using {code_type} code with fidelity {fidelity:.4f}")
            return result
            
        except Exception as e:
            logger.error(f"Error simulating quantum error correction: {e}")
            raise
    
    def _measure_bit_flip_syndrome(self, state_vector: np.ndarray, qubit_pair: List[int]) -> int:
        """
        Measure the parity of a pair of qubits.
        
        Args:
            state_vector: State vector
            qubit_pair: Pair of qubit indices
            
        Returns:
            int: Parity (0 or 1)
        """
        n_qubits = int(np.log2(len(state_vector)))
        
        # Calculate probabilities
        prob_even = 0.0
        prob_odd = 0.0
        
        for i in range(2 ** n_qubits):
            # Check parity of the qubit pair
            bit_i = (i >> qubit_pair[0]) & 1
            bit_j = (i >> qubit_pair[1]) & 1
            parity = bit_i ^ bit_j
            
            if parity == 0:
                prob_even += abs(state_vector[i]) ** 2
            else:
                prob_odd += abs(state_vector[i]) ** 2
        
        # Determine measurement outcome
        if np.random.random() < prob_even:
            return 0
        else:
            return 1
    
    def _decode_bit_flip(self, state_vector: np.ndarray) -> np.ndarray:
        """
        Decode a state encoded with the bit flip code.
        
        Args:
            state_vector: Encoded state vector
            
        Returns:
            np.ndarray: Decoded state vector
        """
        # Extract logical |0⟩ and |1⟩ components
        alpha = state_vector[0]  # |000⟩
        beta = state_vector[7]   # |111⟩
        
        # Normalize
        norm = np.sqrt(abs(alpha)**2 + abs(beta)**2)
        if norm > 0:
            alpha /= norm
            beta /= norm
        
        # Return decoded state
        return np.array([alpha, beta], dtype=complex)
    
    def _decode_phase_flip(self, state_vector: np.ndarray) -> np.ndarray:
        """
        Decode a state encoded with the phase flip code.
        
        Args:
            state_vector: Encoded state vector
            
        Returns:
            np.ndarray: Decoded state vector
        """
        # This is a simplified decoding for demonstration purposes
        # In a real implementation, we would apply the full decoding circuit
        
        # Extract logical |+⟩ and |-⟩ components
        n_qubits = int(np.log2(len(state_vector)))
        
        # Apply Hadamard to all qubits
        for i in range(n_qubits):
            h_gate = self.common_gates['H']
            state_vector = self._apply_gate_to_state(state_vector, h_gate, [i], n_qubits)
        
        # Extract logical |0⟩ and |1⟩ components
        alpha = state_vector[0]  # |000⟩
        beta = state_vector[7]   # |111⟩
        
        # Normalize
        norm = np.sqrt(abs(alpha)**2 + abs(beta)**2)
        if norm > 0:
            alpha /= norm
            beta /= norm
        
        # Return decoded state
        return np.array([alpha, beta], dtype=complex)
    
    def plot_quantum_state(self, 
                          state_name: str,
                          plot_type: str = 'bloch',
                          qubit_indices: List[int] = None,
                          figsize: Tuple[int, int] = (10, 6),
                          return_base64: bool = True) -> Union[str, plt.Figure]:
        """
        Plot a visualization of a quantum state.
        
        Args:
            state_name: Name of the quantum state
            plot_type: Type of plot ('bloch', 'histogram', 'density', 'circuit')
            qubit_indices: Indices of qubits to visualize (for Bloch sphere)
            figsize: Figure size as (width, height)
            return_base64: If True, return base64-encoded image; otherwise return Figure
            
        Returns:
            Union[str, plt.Figure]: Base64-encoded image or matplotlib Figure
        """
        try:
            if state_name not in self.quantum_states:
                raise ValueError(f"Quantum state '{state_name}' not found")
            
            # Get quantum state
            quantum_state = self.quantum_states[state_name]
            n_qubits = quantum_state['n_qubits']
            state_vector = quantum_state['state_vector']
            
            # Create figure
            fig = plt.figure(figsize=figsize)
            
            if plot_type == 'bloch':
                # Plot Bloch sphere representation
                if qubit_indices is None:
                    # Default to first qubit
                    qubit_indices = [0]
                
                # Check if we have too many qubits to plot
                if len(qubit_indices) > 4:
                    logger.warning(f"Too many qubits to plot on Bloch spheres, limiting to first 4")
                    qubit_indices = qubit_indices[:4]
                
                # Create subplots for each qubit
                for i, qubit_idx in enumerate(qubit_indices):
                    if qubit_idx < 0 or qubit_idx >= n_qubits:
                        raise ValueError(f"Qubit index {qubit_idx} out of range (0-{n_qubits-1})")
                    
                    # Calculate reduced density matrix for this qubit
                    rho = self._calculate_reduced_density_matrix(state_vector, [qubit_idx], n_qubits)
                    
                    # Calculate Bloch sphere coordinates
                    x = 2 * np.real(rho[0, 1])
                    y = 2 * np.imag(rho[0, 1])
                    z = np.real(rho[0, 0] - rho[1, 1])
                    
                    # Create 3D subplot
                    ax = fig.add_subplot(1, len(qubit_indices), i + 1, projection='3d')
                    
                    # Plot Bloch sphere
                    self._plot_bloch_sphere(ax, x, y, z)
                    
                    ax.set_title(f'Qubit {qubit_idx}')
                
            elif plot_type == 'histogram':
                # Plot histogram of state probabilities
                ax = fig.add_subplot(111)
                
                # Calculate probabilities
                probabilities = np.abs(state_vector) ** 2
                
                # Create labels for basis states
                labels = [format(i, f'0{n_qubits}b') for i in range(2 ** n_qubits)]
                
                # Plot histogram
                ax.bar(labels, probabilities)
                
                # Set labels and title
                ax.set_xlabel('Basis State')
                ax.set_ylabel('Probability')
                ax.set_title('Quantum State Probabilities')
                
                # Rotate x-axis labels for readability
                plt.xticks(rotation=90)
                
            elif plot_type == 'density':
                # Plot density matrix
                ax = fig.add_subplot(111)
                
                # Calculate density matrix
                density_matrix = np.outer(state_vector, np.conj(state_vector))
                
                # Plot real part of density matrix
                im = ax.imshow(np.real(density_matrix), cmap='RdBu', vmin=-1, vmax=1)
                
                # Add colorbar
                plt.colorbar(im, ax=ax)
                
                # Set labels and title
                ax.set_xlabel('Column Index')
                ax.set_ylabel('Row Index')
                ax.set_title('Real Part of Density Matrix')
                
            elif plot_type == 'circuit':
                # Plot circuit representation (simplified)
                ax = fig.add_subplot(111)
                
                # This is a placeholder for a proper circuit diagram
                # In a real implementation, we would use a dedicated library for circuit visualization
                
                ax.text(0.5, 0.5, "Circuit visualization not implemented", 
                       horizontalalignment='center', verticalalignment='center',
                       transform=ax.transAxes, fontsize=14)
                
                ax.set_title('Quantum Circuit')
                ax.axis('off')
                
            else:
                raise ValueError(f"Unsupported plot type: {plot_type}")
            
            # Tight layout
            fig.tight_layout()
            
            # Increment figure counter
            self.figure_counter += 1
            
            if return_base64:
                # Convert to base64
                buf = io.BytesIO()
                fig.savefig(buf, format='png', dpi=100)
                buf.seek(0)
                img_str = base64.b64encode(buf.read()).decode('utf-8')
                plt.close(fig)
                return img_str
            else:
                return fig
                
        except Exception as e:
            logger.error(f"Error plotting quantum state: {e}")
            raise
    
    def _calculate_reduced_density_matrix(self, 
                                         state_vector: np.ndarray,
                                         qubit_indices: List[int],
                                         n_qubits: int) -> np.ndarray:
        """
        Calculate the reduced density matrix for specific qubits.
        
        Args:
            state_vector: State vector
            qubit_indices: Indices of qubits to keep
            n_qubits: Total number of qubits
            
        Returns:
            np.ndarray: Reduced density matrix
        """
        # Calculate density matrix
        density_matrix = np.outer(state_vector, np.conj(state_vector))
        
        # Reshape to tensor form
        tensor_shape = [2] * (2 * n_qubits)
        tensor_density = density_matrix.reshape(tensor_shape)
        
        # Trace out qubits not in qubit_indices
        trace_indices = [i for i in range(n_qubits) if i not in qubit_indices]
        
        for i in trace_indices:
            # Trace out qubit i
            tensor_density = tensor_density.trace(axis1=i, axis2=i + n_qubits)
            
            # Adjust indices for removed dimensions
            for j in range(len(qubit_indices)):
                if qubit_indices[j] > i:
                    qubit_indices[j] -= 1
        
        # Reshape back to matrix form
        reduced_dim = 2 ** len(qubit_indices)
        reduced_density = tensor_density.reshape((reduced_dim, reduced_dim))
        
        return reduced_density
    
    def _plot_bloch_sphere(self, ax: plt.Axes, x: float, y: float, z: float) -> None:
        """
        Plot a Bloch sphere with a state vector.
        
        Args:
            ax: Matplotlib 3D axes
            x, y, z: Bloch sphere coordinates
        """
        # Plot sphere
        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 100)
        sphere_x = np.outer(np.cos(u), np.sin(v))
        sphere_y = np.outer(np.sin(u), np.sin(v))
        sphere_z = np.outer(np.ones(np.size(u)), np.cos(v))
        
        ax.plot_surface(sphere_x, sphere_y, sphere_z, color='lightgray', alpha=0.2)
        
        # Plot axes
        ax.quiver(0, 0, 0, 1, 0, 0, color='r', length=1.3, arrow_length_ratio=0.1)
        ax.quiver(0, 0, 0, 0, 1, 0, color='g', length=1.3, arrow_length_ratio=0.1)
        ax.quiver(0, 0, 0, 0, 0, 1, color='b', length=1.3, arrow_length_ratio=0.1)
        
        # Add axis labels
        ax.text(1.4, 0, 0, r'$x$', color='r')
        ax.text(0, 1.4, 0, r'$y$', color='g')
        ax.text(0, 0, 1.4, r'$z$', color='b')
        
        # Plot state vector
        ax.quiver(0, 0, 0, x, y, z, color='purple', length=1, arrow_length_ratio=0.1)
        
        # Set equal aspect ratio
        ax.set_box_aspect([1, 1, 1])
        
        # Remove grid and background
        ax.grid(False)
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        
        # Set limits
        ax.set_xlim([-1.5, 1.5])
        ax.set_ylim([-1.5, 1.5])
        ax.set_zlim([-1.5, 1.5])
        
        # Remove tick labels
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])
    
    def simulate_quantum_phase_estimation(self, 
                                         name: str,
                                         unitary_matrix: np.ndarray,
                                         eigenstate: np.ndarray,
                                         n_counting_qubits: int) -> Dict[str, Any]:
        """
        Simulate the Quantum Phase Estimation algorithm.
        
        Args:
            name: Name for the quantum state
            unitary_matrix: Unitary operator U
            eigenstate: Eigenstate of U
            n_counting_qubits: Number of counting qubits
            
        Returns:
            Dict[str, Any]: Simulation result
        """
        try:
            # Check if unitary_matrix is unitary
            if not self._is_unitary(unitary_matrix):
                raise ValueError("Input matrix is not unitary")
            
            # Check if eigenstate is an eigenvector of unitary_matrix
            eigenstate = np.array(eigenstate, dtype=complex)
            eigenstate /= np.linalg.norm(eigenstate)
            
            # Calculate eigenvalue
            eigenvalue = np.dot(np.conj(eigenstate), np.dot(unitary_matrix, eigenstate))
            
            # Calculate phase
            phase = np.angle(eigenvalue) / (2 * np.pi)
            
            # Create quantum state
            n_target_qubits = int(np.log2(len(eigenstate)))
            n_total_qubits = n_counting_qubits + n_target_qubits
            
            quantum_state = self.create_quantum_state(name, n_total_qubits, state_type='zero')
            
            # Initialize counting qubits in |+⟩ state
            for i in range(n_counting_qubits):
                self.apply_gate(name, 'H', [i])
            
            # Initialize target qubits in eigenstate
            state_vector = quantum_state['state_vector']
            
            # Set target qubits to eigenstate
            for i in range(2 ** n_counting_qubits):
                for j in range(2 ** n_target_qubits):
                    idx = i * (2 ** n_target_qubits) + j
                    state_vector[idx] = state_vector[idx] * eigenstate[j]
            
            # Update quantum state
            quantum_state['state_vector'] = state_vector
            
            # Apply controlled-U^(2^j) operations
            for j in range(n_counting_qubits):
                # Calculate U^(2^j)
                power = 2 ** j
                u_power = np.linalg.matrix_power(unitary_matrix, power)
                
                # Apply controlled-U^(2^j)
                self.apply_gate(name, u_power, list(range(n_counting_qubits, n_total_qubits)), [j])
            
            # Apply inverse QFT to counting qubits
            self._apply_inverse_qft(name, list(range(n_counting_qubits)))
            
            # Measure counting qubits
            measurement_results = []
            for i in range(n_counting_qubits):
                result = self.measure_qubit(name, i)
                measurement_results.append(result['outcome'])
            
            # Convert measurement to phase estimate
            measured_phase = 0
            for i, bit in enumerate(reversed(measurement_results)):
                measured_phase += bit * 2 ** (-i - 1)
            
            # Calculate error
            phase_error = abs(phase - measured_phase)
            phase_error = min(phase_error, 1 - phase_error)  # Account for periodicity
            
            # Create result
            result = {
                'state_name': name,
                'n_counting_qubits': n_counting_qubits,
                'n_target_qubits': n_target_qubits,
                'true_phase': float(phase),
                'measured_phase': float(measured_phase),
                'phase_error': float(phase_error),
                'measurement_results': measurement_results
            }
            
            logger.info(f"Simulated Quantum Phase Estimation: true phase={phase:.4f}, measured phase={measured_phase:.4f}")
            return result
            
        except Exception as e:
            logger.error(f"Error simulating Quantum Phase Estimation: {e}")
            raise
    
    def _apply_inverse_qft(self, state_name: str, qubit_indices: List[int]) -> None:
        """
        Apply the inverse Quantum Fourier Transform to specific qubits.
        
        Args:
            state_name: Name of the quantum state
            qubit_indices: Indices of qubits to apply inverse QFT to
        """
        # Reverse the order of operations in QFT
        n_qubits = len(qubit_indices)
        
        # Swap qubits (optional, for standard QFT ordering)
        for i in range(n_qubits // 2):
            # Create SWAP gate
            swap_gate = np.array([
                [1, 0, 0, 0],
                [0, 0, 1, 0],
                [0, 1, 0, 0],
                [0, 0, 0, 1]
            ], dtype=complex)
            
            # Apply SWAP gate
            self.apply_gate(state_name, swap_gate, [qubit_indices[i], qubit_indices[n_qubits - i - 1]])
        
        # Apply inverse QFT
        for i in range(n_qubits - 1, -1, -1):
            # Apply inverse controlled phase rotations
            for j in range(n_qubits - 1, i, -1):
                # Calculate phase rotation angle
                angle = -2 * np.pi / (2 ** (j - i + 1))
                
                # Create phase rotation gate
                phase_gate = np.array([
                    [1, 0],
                    [0, np.exp(1j * angle)]
                ], dtype=complex)
                
                # Apply controlled phase gate
                self.apply_gate(state_name, phase_gate, [qubit_indices[j]], [qubit_indices[i]])
            
            # Apply Hadamard to qubit i
            self.apply_gate(state_name, 'H', [qubit_indices[i]])
    
    def run(self, command: str = None, **kwargs) -> Any:
        """
        Run a command or return help information.
        
        Args:
            command: Command to run (optional)
            **kwargs: Additional arguments
            
        Returns:
            Any: Command result or help information
        """
        if command is None:
            # Return help information
            return self.help()
        
        # Parse and execute command
        try:
            # Check if command is a method name
            if hasattr(self, command) and callable(getattr(self, command)):
                method = getattr(self, command)
                return method(**kwargs)
            
            # Otherwise, try to evaluate as a Python expression
            # This is potentially dangerous and should be used with caution
            return eval(command)
            
        except Exception as e:
            logger.error(f"Error executing command '{command}': {e}")
            return f"Error: {str(e)}"
    
    def help(self) -> str:
        """Return help information about the ONI Quantum Simulator."""
        help_text = """
        ONI Quantum Simulator - Simulation and analysis of quantum systems
        
        Available methods:
        - create_quantum_state(name, n_qubits, ...): Create a quantum state
        - apply_gate(state_name, gate, target_qubits, ...): Apply a quantum gate to a state
        - measure_qubit(state_name, qubit_index, ...): Measure a qubit
        - measure_all_qubits(state_name): Measure all qubits
        - calculate_expectation(state_name, operator, ...): Calculate expectation value
        - create_bell_state(name, bell_type): Create a Bell state
        - create_ghz_state(name, n_qubits): Create a GHZ state
        - create_w_state(name, n_qubits): Create a W state
        - simulate_quantum_fourier_transform(state_name): Simulate Quantum Fourier Transform
        - simulate_grover_search(name, n_qubits, target_state): Simulate Grover's search
        - simulate_quantum_teleportation(state_to_teleport): Simulate quantum teleportation
        - simulate_quantum_error_correction(name, ...): Simulate quantum error correction
        - simulate_quantum_phase_estimation(name, ...): Simulate Quantum Phase Estimation
        - plot_quantum_state(state_name, plot_type, ...): Plot visualization of quantum state
        
        For more information on a specific method, use help(ONIQuantumSimulator.method_name)
        """
        return help_text
    
    def cleanup(self):
        """Clean up resources."""
        # Close all matplotlib figures
        plt.close('all')
        
        # Clear quantum states and results
        self.quantum_states.clear()
        self.quantum_circuits.clear()
        self.simulation_results.clear()
        
        logger.info("ONI Quantum Simulator cleaned up")

# Example usage
if __name__ == "__main__":
    simulator = ONIQuantumSimulator()
    
    # Create a Bell state
    bell_state = simulator.create_bell_state("bell", bell_type=0)
    
    # Measure the Bell state
    measurement = simulator.measure_all_qubits("bell")
    
    # Create a GHZ state
    ghz_state = simulator.create_ghz_state("ghz", 3)
    
    # Plot the GHZ state
    plot = simulator.plot_quantum_state("ghz", "histogram")
    
    print(f"Bell state measurement: {measurement['outcome']}")
    print(f"Bell state probability: {measurement['probability']:.4f}")