from abc import ABC, abstractmethod

from ase import Atoms
from ase.calculators.calculator import Calculator
from numpy import nan
from pydantic import BaseModel

__all__ = [
    "MonteCarloMove",
    "MonteCarloRecord",
    "MonteCarloState",
]


class MonteCarloState(Atoms):
    _state_keys: set[str] = set()

    def __init__(
        self,
        symbols=None,
        positions=None,
        numbers=None,
        tags=None,
        momenta=None,
        masses=None,
        magmoms=None,
        charges=None,
        scaled_positions=None,
        cell=None,
        pbc=None,
        celldisp=None,
        constraint=None,
        calculator=None,
        info=None,
        velocities=None,
    ) -> None:
        super().__init__(
            symbols,
            positions,
            numbers,
            tags,
            momenta,
            masses,
            magmoms,
            charges,
            scaled_positions,
            cell,
            pbc,
            celldisp,
            constraint,
            calculator,
            info,
            velocities,
        )
        for k in self._state_keys:
            v = self.info.get(k, None)
            if v is None:
                raise KeyError("The key of {k} not found in Atoms.info")
            else:
                self.info[k] = v


class MonteCarloRecord(BaseModel):
    """A class to store the record of the Monte Carlo move."""

    accept: bool = False
    energy_old: float = nan
    energy_new: float = nan
    energy_change: float = nan
    boltzmann_factor: float = nan
    random_number: float = nan
    stepsize: float = nan
    fmax_old: float = nan
    fmax_new: float = nan


class MonteCarloMove(ABC):
    def __init__(
        self,
        state: MonteCarloState,
        calculator: Calculator | None = None,
    ) -> None:
        if not isinstance(calculator, Calculator):
            if not isinstance(state.calc, Calculator):
                raise ValueError("The calculator is not set")
            calculator = state.calc
            calculator.results.clear()
        self.calc = calculator
        self.state = state

    @abstractmethod
    def _get_new_atoms(self, stepsize: float) -> Atoms: ...
    @abstractmethod
    def __call__(
        self, *args, **kwds
    ) -> tuple[MonteCarloState, MonteCarloRecord]: ...

    @classmethod
    def run(
        cls,
        state: MonteCarloState,
        calculator: Calculator | None = None,
        *args,
        **kwds,
    ) -> tuple[MonteCarloState, MonteCarloRecord]:
        return cls(state, calculator).__call__(*args, **kwds)
