from abc import abstractmethod
from dataclasses import dataclass, fields
from datetime import datetime
from pathlib import Path
from typing import Any, Self

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.calculator import Calculator
from ase.calculators.emt import EMT
from ase.db.row import AtomsRow, atoms2dict
from ase.io import read
from ase.io.trajectory import Trajectory
from ase.optimize.optimize import Dynamics
from typing_extensions import override


@dataclass(slots=True)
class SimState:
    atoms: Atoms

    @abstractmethod
    def __post_init__(self) -> None:
        assert isinstance(self.atoms, Atoms), (
            "The atoms must be a instance of ASE atoms."
            + f" But got {type(self.atoms)}."
        )

    @classmethod
    def from_ase(cls, atoms: Atoms, **kwargs) -> Self:
        """Create a state from an ASE atoms."""
        dct = {}
        for f in fields(cls):
            if f.name in kwargs:
                dct[f.name] = kwargs[f.name]
            elif f.name in atoms.info:
                dct[f.name] = atoms.info[f.name]
            else:
                pass
        return cls(atoms=atoms, **dct)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        """Create a state from a dict."""
        row = AtomsRow(data)
        atoms = row.toatoms()
        ks = (k for k in row)
        dct = {k: v for k, v in data.items() if k in ks}
        return cls.from_ase(atoms, **dct)

    def to_dict(self) -> dict[str, Any]:
        """Convert the state to a dict."""
        dct, exclude_keys = atoms2dict(self.atoms), ("unique_id", "atoms")
        dct.update({f.name: getattr(self, f.name) for f in fields(self)})
        return {k: v for k, v in dct.items() if k not in exclude_keys}

    def to_ase(self) -> Atoms:
        """Convert the state to an ASE atoms."""
        return Atoms(
            self.atoms,
            info={
                f.name: getattr(self, f.name)
                for f in fields(self)
                if f.name not in ("atoms",)
            },
        )


class SimABC(Dynamics):
    def __init__(
        self,
        *,
        calculator: str | Calculator = "emt",
        atoms: Path | str | None = None,
        record_interval: int = 1,
        record_file: str | Path = "record.table",
        trajectory: str | Path = "ase.traj",
        trajectory_interval: int = 1,
        max_steps: int = 100_000_000,
        workdir: Any | str | Path = ".",
    ) -> None:
        """The abstract class for simulation.

        Args:
            calculator (str | Calculator, optional): The SPE method
                for simulation. It's format like
                NAME,ARG0,ARG1,ARG...,KWARG0=VAL0,KWARG1=VAL1,...
                See details for the function of `get_calculator`.
                Defaults to "emt".
            atoms (Path | str | None, optional): The file to read the structure
                from. If the value is None, the structure will be read from
                either 'POSCAR' or 'structure.xyz'. Defaults to None.
            record_interval (int, optional): Only write record line
                for every *interval* time steps. Defaults to 1.
            record_file (str | Path, optional): The file to record something.
                Typically, the record file is a table of the simulation
                results and it can be read as a pandas dataframe.
                Defaults to "record.table".
            trajectory (str | Path, optional): The file to
                write the trajectory. Defaults to "ase.traj".
            trajectory_interval (int, optional): Only write trajectory
                for every *interval* time steps. Defaults to 1.
            max_steps (int, optional): The maximum number
                of simulation steps. Defaults to 100_000_000.
            workdir (str | Path, optional): The work folder. Defaults to ".".
        """
        print(type(workdir), workdir)
        self._workdir = Path(workdir)
        self._workdir.mkdir(parents=True, exist_ok=True)
        self._recordfile = self._workdir.joinpath(Path(record_file).name)
        self._trajfile = self._workdir.joinpath(Path(trajectory).name)
        self._max_steps = int(max_steps)
        if not isinstance(atoms, Atoms):
            if isinstance(atoms, str | Path):
                atoms = read(Path(atoms), index=-1)  # type: ignore
            else:
                for fname in ["structure.xyz", "POSCAR"]:
                    p = self._workdir.joinpath(Path(fname).name)
                    atoms = read(p, index=-1)  # type: ignore
                    if isinstance(atoms, Atoms):
                        break
            assert isinstance(atoms, Atoms), (
                "Put your 'structure.xyz' or 'POSCAR' here, "
                "or set the file path in configuration file."
            )
        atoms.calc = (
            self.get_calculator(calculator)  # noqa: F821
            if not isinstance(calculator, Calculator)
            else calculator
        )
        super().__init__(
            atoms,
            self._recordfile,  # type: ignore
            loginterval=int(record_interval),
        )
        assert isinstance(self.atoms, Atoms)
        self.attach(
            self.closelater(Trajectory(self._trajfile, mode="w")),
            interval=int(trajectory_interval),
            atoms=self.optimizable,
        )

    @classmethod
    def get_calculator(cls, method: str) -> Calculator:
        args, kwargs = [], {}
        if method.count(",") != 0:
            split = method.split(",")
            for arg in split[1:]:
                match arg.count("="):
                    case 0:
                        args.append(arg)
                    case 1:
                        k, v = arg.split("=")
                        kwargs[k] = v
                    case _:
                        raise KeyError(f"Invalid SPE method: {method}")
            method = split[0]

        method = method.lower()
        try:
            if method == "emt":
                result = EMT(*args, **kwargs)
        except Exception as e:
            raise KeyError(f"Invalid SPE method: {method}; {e}")
        return result  # type: ignore

    @override
    def todict(self) -> dict[str, str]:
        return {}

    @override
    def irun(self, steps=100_000_000) -> Any:
        # update the maximum number of steps
        self.max_steps = self.nsteps + steps

        # log the initial step
        if self.nsteps == 0:
            self.log()
            self.call_observers()

        # check convergence
        is_converged = False
        yield is_converged

        # run the algorithm until converged or max_steps reached
        while not is_converged and self.nsteps < self.max_steps:
            # compute the next step
            self.step()
            self.nsteps += 1

            # log the step
            self.log()
            self.call_observers()

            # check convergence
            is_converged = False
            yield is_converged

    def run(self) -> Any:  # type: ignore
        for converged in self.irun(self._max_steps):
            pass
        return converged  # type: ignore

    @abstractmethod
    @override
    def step(self) -> None: ...  # type: ignore

    @abstractmethod
    @override
    def log(self, *args, **kwargs) -> None:  # type: ignore
        if self.logfile is not None:
            if self.nsteps == 0:
                lst: list[str] = [
                    f"{self.__class__.__name__}:",
                    f"{'TIME':<24s}",
                    f"{'STEP':>8s}",
                ] + self.log_header()
                self.logfile.write(f"{'  '.join(lst)}\n")

            T = datetime.now()
            lst: list[str] = [
                f"{self.__class__.__name__}:",
                f"{T.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]:<24s}",
                f"{self.nsteps:8d}",
            ] + self.log_lst()
            self.logfile.write(f"{'  '.join(lst)}\n")
            self.logfile.flush()
            self.logfile.flush()

    def log_header(self) -> list[str]:
        return [f"{'E':>10s}", f"{'Fmax':>10s}"]

    def log_lst(self) -> list[str]:
        f = self.atoms.get_forces()
        e = self.atoms.get_potential_energy()
        fmax = np.sqrt((f**2).sum(axis=1).max())
        return [f"{e:10.4f}", f"{fmax:10.4f}"]


@pytest.mark.parametrize(
    "spe",
    [
        "emt",
        # "gfnff",
        # "xtb,gfn1xtb",
        # "xtb,method=gfn2xtb",
    ],
)
def test_get_calculator(spe: str) -> None:
    print(SimABC.get_calculator(spe))


if __name__ == "__main__":
    pytest.main([__file__, "-s", "-v", "--maxfail=5"])
