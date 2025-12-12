from typing import override

import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator
from ase.geometry.geometry import find_mic
from ase.units import kB

from ._base import MonteCarloMove as Move
from ._base import MonteCarloRecord as Record
from ._state import CanonicalState


class SingleParticleDisplacement(Move):
    def __init__(
        self,
        state: CanonicalState,
        calculator: Calculator | None = None,
    ) -> None:
        assert isinstance(state, CanonicalState), (
            "The state must be instance of CanonicalState."
        )
        super().__init__(state, calculator)

    @override
    def _get_new_atoms(self, stepsize: float) -> tuple[Atoms, int]:  # type: ignore
        i = np.random.randint(len(self.state))
        dpos = np.zeros_like(self.state.positions)
        dpos_i = np.random.randn(3)
        dpos_i /= np.linalg.norm(dpos_i)
        dpos[i, :] += dpos_i * float(stepsize)
        return Atoms(
            self.state,
            self.state.positions + dpos,
            calculator=self.calc,
        ), i

    def _complete_record(
        self,
        record: Record,
        **kwargs,
    ) -> tuple[Record, Atoms]:
        assert not np.isnan(record.energy_old)
        assert not np.isnan(record.fmax_old)
        assert not np.isnan(record.stepsize)
        assert not np.isnan(record.random_number)
        new_atoms, _ = self._get_new_atoms(record.stepsize)
        record.energy_new = e1 = new_atoms.get_potential_energy()
        f1: np.ndarray = new_atoms.get_forces()
        record.fmax_new = np.max(np.linalg.norm(f1, axis=1))
        record.energy_change = e1 - record.energy_old
        return record, new_atoms

    @override
    def __call__(
        self,
        stepsize_min: float = 0.1,
        stepsize_max: float = 0.5,
        temperature: float = 300.0,
        **kwargs,
    ) -> tuple[CanonicalState, Record]:
        e0 = float(self.state.info.get("U"))  # type: ignore
        try:
            assert isinstance(self.state.calc, Calculator)
            f0: np.ndarray = self.state.calc.results.get("forces")  # type: ignore
            fmax0 = np.max(np.linalg.norm(f0, axis=1))
        except Exception:
            fmax0 = np.nan
        stepsize: float = np.random.uniform(
            float(stepsize_min),
            float(stepsize_max),
        )
        record, new_atoms = self._complete_record(
            Record(
                energy_old=e0,
                fmax_old=fmax0,
                random_number=np.random.rand(),
                stepsize=stepsize,
            ),
            **kwargs,
        )
        assert not np.isnan(record.energy_change)
        assert not np.isnan(record.energy_new)
        assert not np.isnan(record.fmax_new)

        kBT = kB * abs(float(temperature))
        record.boltzmann_factor = min(1, np.exp(-record.energy_change / kBT))
        record.accept = accept = record.random_number < record.boltzmann_factor
        new_atoms.info.update({"U": e0 + record.energy_change})
        if accept:
            return CanonicalState(new_atoms), record
        else:
            assert isinstance(self.state, CanonicalState)
            return self.state, record

    @classmethod
    def run(  # type: ignore
        cls,
        state: CanonicalState,
        calculator: Calculator | None = None,
        stepsize_min: float = 0.1,
        stepsize_max: float = 0.5,
        temperature: float = 300.0,
    ) -> tuple[CanonicalState, Record]:
        return cls(state, calculator).__call__(
            stepsize_min=stepsize_min,
            stepsize_max=stepsize_max,
            temperature=temperature,
        )


class SingleParticleDisplacementCutoff(SingleParticleDisplacement):
    @override
    def _complete_record(
        self,
        record: Record,
        cutoff: float = np.inf,
        **kwargs,
    ) -> tuple[Record, Atoms]:
        if np.isinf(cutoff):
            return super()._complete_record(record)
        else:
            assert not np.isnan(record.energy_old)
            assert not np.isnan(record.fmax_old)
            assert not np.isnan(record.stepsize)
            assert not np.isnan(record.random_number)
            new_atoms, i = self._get_new_atoms(record.stepsize)
            v = new_atoms.positions - new_atoms.positions[i]
            _, dmic = find_mic(v, new_atoms.cell, True)
            nbrs = np.where(dmic < float(cutoff))[0]

            atoms0: Atoms = self.state.__getitem__(nbrs)  # type: ignore
            atoms0.calc = self.calc
            atoms0.calc.results = {}  # type: ignore
            e0 = atoms0.get_potential_energy()
            f0 = atoms0.get_forces()
            record.fmax_old = np.max(np.linalg.norm(f0, axis=1))

            atoms1: Atoms = new_atoms.__getitem__(nbrs)  # type: ignore
            atoms1.calc = new_atoms.calc
            atoms1.calc.results = {}  # type: ignore
            e1 = atoms1.get_potential_energy()
            f1 = atoms1.get_forces()
            record.fmax_new = np.max(np.linalg.norm(f1, axis=1))
            record.energy_change = e1 - e0
            return record, new_atoms

    @classmethod
    def run(  # type: ignore
        cls,
        state: CanonicalState,
        calculator: Calculator | None = None,
        stepsize_min: float = 0.1,
        stepsize_max: float = 0.5,
        temperature: float = 300.0,
        cutoff: float = np.inf,
    ) -> tuple[CanonicalState, Record]:
        return cls(state, calculator).__call__(
            stepsize_min=stepsize_min,
            stepsize_max=stepsize_max,
            temperature=temperature,
            cutoff=cutoff,
        )
