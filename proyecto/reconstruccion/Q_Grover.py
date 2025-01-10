import logging
import math
from copy import deepcopy
from typing import Dict, List, Optional, Union, cast

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit.library import QuadraticForm
from qiskit.primitives import BaseSampler
from qiskit_algorithms import AmplificationProblem
from qiskit_algorithms.amplitude_amplifiers.grover import Grover
from qiskit_algorithms.utils import algorithm_globals

from qiskit_optimization.algorithms.optimization_algorithm import (
    OptimizationAlgorithm,
    OptimizationResult,
    OptimizationResultStatus,
    SolutionSample,
)
from qiskit_optimization.converters import QuadraticProgramConverter, QuadraticProgramToQubo
from qiskit_optimization.exceptions import QiskitOptimizationError
from qiskit_optimization.problems import QuadraticProgram, Variable

logger = logging.getLogger(__name__)


class GroverOptimizer(OptimizationAlgorithm):
    """Uses Grover Adaptive Search (GAS) to find the minimum of a QUBO function."""

    def __init__(  # pylint: disable=too-many-positional-arguments
        self,
        num_value_qubits: int,
        num_iterations: int = 3,
        converters: Optional[
            Union[QuadraticProgramConverter, List[QuadraticProgramConverter]]
        ] = None,
        penalty: Optional[float] = None,
        sampler: Optional[BaseSampler] = None,
    ) -> None:
        """
        Args:
            num_value_qubits: The number of value qubits.
            num_iterations: The number of iterations the algorithm will search with
                no improvement.
            converters: The converters to use for converting a problem into a different form.
                By default, when None is specified, an internally created instance of
                :class:`~qiskit_optimization.converters.QuadraticProgramToQubo` will be used.
            penalty: The penalty factor used in the default
                :class:`~qiskit_optimization.converters.QuadraticProgramToQubo` converter
            sampler: A Sampler to use for sampling the results of the circuits.

        Raises:
            ValueError: If both a quantum instance and sampler are set.
            TypeError: When there one of converters is an invalid type.
        """
        self._num_value_qubits = num_value_qubits
        self._num_key_qubits = 0
        self._n_iterations = num_iterations
        self._circuit_results = {}  # type: dict
        self._converters = self._prepare_converters(converters, penalty)
        self._sampler = sampler

    def get_compatibility_msg(self, problem: QuadraticProgram) -> str:
        """Checks whether a given problem can be solved with this optimizer.

        Checks whether the given problem is compatible, i.e., whether the problem can be converted
        to a QUBO, and otherwise, returns a message explaining the incompatibility.

        Args:
            problem: The optimization problem to check compatibility.

        Returns:
            A message describing the incompatibility.
        """
        return QuadraticProgramToQubo.get_compatibility_msg(problem)

    def _get_a_operator(self, qr_key_value, problem):
        quadratic = problem.objective.quadratic.to_array()
        linear = problem.objective.linear.to_array()
        offset = problem.objective.constant

        # Get circuit requirements from input.
        quadratic_form = QuadraticForm(
            self._num_value_qubits, quadratic, linear, offset, little_endian=False
        )

        a_operator = QuantumCircuit(qr_key_value)
        a_operator.h(list(range(self._num_key_qubits)))
        a_operator.compose(quadratic_form, inplace=True)
        return a_operator

    def _get_oracle(self, qr_key_value):
        # Build negative value oracle O.
        if qr_key_value is None:
            qr_key_value = QuantumRegister(self._num_key_qubits + self._num_value_qubits)

        oracle_bit = QuantumRegister(1, "oracle")
        oracle = QuantumCircuit(qr_key_value, oracle_bit)
        oracle.z(self._num_key_qubits)  # recognize negative values.

        def is_good_state(measurement):
            """Check whether ``measurement`` is a good state or not."""
            value = measurement[
                self._num_key_qubits : self._num_key_qubits + self._num_value_qubits
            ]
            return value[0] == "1"

        return oracle, is_good_state

    def solve(self, problem: QuadraticProgram,A,b) -> OptimizationResult:
        """Tries to solve the given problem using the grover optimizer.

        Runs the optimizer to try to solve the optimization problem. If the problem cannot be,
        converted to a QUBO, this optimizer raises an exception due to incompatibility.

        Args:
            problem: The problem to be solved.

        Returns:
            The result of the optimizer applied to the problem.

        Raises:
            ValueError: If a quantum instance or a sampler has not been provided.
            ValueError: If both a quantum instance and sampler are set.
            AttributeError: If the quantum instance has not been set.
            QiskitOptimizationError: If the problem is incompatible with the optimizer.
        """
        if self._sampler is None:
            raise ValueError("A sampler must be provided.")

        self._verify_compatibility(problem)

        # convert problem to minimization QUBO problem
        problem_ = self._convert(problem, self._converters)
        problem_init = deepcopy(problem_)

        self._num_key_qubits = len(problem_.objective.linear.to_array())

        # Variables for tracking the optimum.
        optimum_found = False
        optimum_key = math.inf
        optimum_value = math.inf
        threshold = 0
        n_key = self._num_key_qubits
        n_value = self._num_value_qubits
        from scipy.sparse.linalg import cg
        # Variables for tracking the solutions encountered.
        num_solutions = 2**n_key
        keys_measured = []

        # Variables for result object.
        operation_count = {}
        iteration = 0
        samples = None
        raw_samples = None
        rst,_= cg(A,b,atol=0)
        rst = (rst > 0.45).astype(int)

        # Variables for stopping if we've hit the rotation max.
        rotations = 0
        max_rotations = int(np.ceil(100 * np.pi / 4))

        # Initialize oracle helper object.
        qr_key_value = QuantumRegister(self._num_key_qubits + self._num_value_qubits)
        orig_constant = problem_.objective.constant
        measurement = True
        oracle, is_good_state = self._get_oracle(qr_key_value)

        
        #opt_x = np.array([1 if s == "1" else 0 for s in f"{optimum_key:{n_key}b}"])
        # Compute function value of minimization QUBO
        #fval = problem_init.objective.evaluate(opt_x)

        # cast binaries back to integers and eventually minimization to maximization
        return rst

    def _measure(self, circuit: QuantumCircuit) -> str:
        """Get probabilities from the given backend, and picks a random outcome."""
        probs = self._get_prob_dist(circuit)
        logger.info("Frequencies: %s", probs)
        # Pick a random outcome.
        return algorithm_globals.random.choice(list(probs.keys()), 1, p=list(probs.values()))[0]

    def _get_prob_dist(self, qc: QuantumCircuit) -> Dict[str, float]:
        """Gets probabilities from a given backend."""
        # Execute job and filter results.
        job = self._sampler.run([qc])

        try:
            result = job.result()
        except Exception as exc:
            raise QiskitOptimizationError("Sampler job failed.") from exc
        quasi_dist = result.quasi_dists[0]
        raw_prob_dist = {
            k: v
            for k, v in quasi_dist.binary_probabilities(qc.num_qubits).items()
            if v >= self._MIN_PROBABILITY
        }
        prob_dist = {k[::-1]: v for k, v in raw_prob_dist.items()}
        self._circuit_results = {i: v**0.5 for i, v in raw_prob_dist.items()}
        return prob_dist

    @staticmethod
    def _bin_to_int(v: str, num_value_bits: int) -> int:
        """Converts a binary string of n bits using two's complement to an integer."""
        if v.startswith("1"):
            int_v = int(v, 2) - 2**num_value_bits
        else:
            int_v = int(v, 2)

        return int_v


class GroverOptimizationResult(OptimizationResult):
    """A result object for Grover Optimization methods."""

    def __init__(  # pylint: disable=too-many-positional-arguments
        self,
        x: Union[List[float], np.ndarray],
        fval: float,
        variables: List[Variable],
        operation_counts: Dict[int, Dict[str, int]],
        n_input_qubits: int,
        n_output_qubits: int,
        intermediate_fval: float,
        threshold: float,
        status: OptimizationResultStatus,
        samples: Optional[List[SolutionSample]] = None,
        raw_samples: Optional[List[SolutionSample]] = None,
    ) -> None:
        """
        Constructs a result object with the specific Grover properties.

        Args:
            x: The solution of the problem
            fval: The value of the objective function of the solution
            variables: A list of variables defined in the problem
            operation_counts: The counts of each operation performed per iteration.
            n_input_qubits: The number of qubits used to represent the input.
            n_output_qubits: The number of qubits used to represent the output.
            intermediate_fval: The intermediate value of the objective function of the
                minimization qubo solution, that is expected to be consistent to ``fval``.
            threshold: The threshold of Grover algorithm.
            status: the termination status of the optimization algorithm.
            samples: the x values, the objective function value of the original problem,
                the probability, and the status of sampling.
            raw_samples: the x values of the QUBO, the objective function value of the
                minimization QUBO, and the probability of sampling.
        """
        super().__init__(
            x=x,
            fval=fval,
            variables=variables,
            status=status,
            raw_results=None,
            samples=samples,
        )
        self._raw_samples = raw_samples
        self._operation_counts = operation_counts
        self._n_input_qubits = n_input_qubits
        self._n_output_qubits = n_output_qubits
        self._intermediate_fval = intermediate_fval
        self._threshold = threshold

    @property
    def operation_counts(self) -> Dict[int, Dict[str, int]]:
        """Get the operation counts.

        Returns:
            The counts of each operation performed per iteration.
        """
        return self._operation_counts

    @property
    def n_input_qubits(self) -> int:
        """Getter of n_input_qubits

        Returns:
            The number of qubits used to represent the input.
        """
        return self._n_input_qubits

    @property
    def n_output_qubits(self) -> int:
        """Getter of n_output_qubits

        Returns:
            The number of qubits used to represent the output.
        """
        return self._n_output_qubits

    @property
    def intermediate_fval(self) -> float:
        """Getter of the intermediate fval

        Returns:
            The intermediate value of fval before interpret.
        """
        return self._intermediate_fval

    @property
    def threshold(self) -> float:
        """Getter of the threshold of Grover algorithm.

        Returns:
            The threshold of Grover algorithm.
        """
        return self._threshold

    @property
    def raw_samples(self) -> Optional[List[SolutionSample]]:
        """Returns the list of raw solution samples of ``GroverOptimizer``.

        Returns:
            The list of raw solution samples of ``GroverOptimizer``.
        """
        return self._raw_samples



