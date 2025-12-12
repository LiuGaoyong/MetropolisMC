from abc import abstractmethod
from pathlib import Path
from time import localtime

import numpy as np
from ase import Atoms
from ase.io import read
from ase.io.trajectory import Trajectory
from ase.optimize.optimize import Dynamics
from typing_extensions import Any, override
from loguru import logger

from ._config import Config
from ._utils import get_calculator


class SimulationABC(Dynamics):
    def __init__(
        self,
        atoms: Atoms | Path | str,
        trajectory: str | Path = "ase.traj",
        recordfile: str | Path = "record.table",
        logfile: str | Path = "-",
        *,
        spe_method: str = "emt",
    ) -> None:
        if isinstance(config, str | Path):
            config = Config.read(Path(config))
        config = Config.model_validate(config)
        self._workdir, self._config = Path(workdir), config
        assert self._workdir.exists() and self._workdir.is_dir()
        assert isinstance(self._config, Config)
        calc = get_calculator(config.simulation_spe)

        trajfile = config.trajectory_file if trajectory is None else trajectory
        record_file = config.record_file if logfile is None else logfile
        if config.restart:
            assert trajfile is not None, (
                "The trajectory file is not set for restart simulation."
            )
            p = self._workdir.joinpath(trajfile)
            assert p.exists(), f"The trajectory file of '{p}' does not exist."
            assert p.is_file(), f"The trajectory file of '{p}' is not file."
            atoms = read(p, index=-1)  # type: ignore
            assert isinstance(atoms, Atoms), (
                "Read atoms from trajectory file of '{p}' fails."
            )
        else:
            if not isinstance(atoms, Atoms):
                for fname in (
                    [str(self._config.structure_file)]
                    if self._config.structure_file is not None
                    else ["structure.xyz", "POSCAR"]
                ):
                    p = self._workdir.joinpath(Path(fname).name)
                    atoms = read(p, index=-1)  # type: ignore
                    if isinstance(atoms, Atoms):
                        break
                assert isinstance(atoms, Atoms), (
                    "Put your 'structure.xyz' or 'POSCAR' here, "
                    "or set the file path in configuration file."
                )
            if record_file is not None:
                self._workdir.joinpath(record_file).unlink(missing_ok=True)

        assert isinstance(atoms, Atoms), (type(atoms), atoms)
        atoms.calc = calc

        super().__init__(atoms, record_file, loginterval=config.record_interval)  # type: ignore
        assert isinstance(self.atoms, Atoms)
        if trajfile is not None:
            self.attach(
                self.closelater(
                    Trajectory(
                        self._workdir.joinpath(Path(trajfile).name),
                        mode="a" if config.restart else "w",
                    )
                ),
                interval=config.trajectory_interval,
                atoms=self.optimizable,
            )

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

    def run(self, steps=100_000_000) -> Any:
        for converged in self.irun(steps):
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
                    f"{'TIME':>10s}",
                    f"{'STEP':>8s}",
                    f"{'E':>10s}",
                    f"{'Fmax':>10s}",
                ]
                self.logfile.write(f"{'  '.join(lst)}\n")

            T = localtime()
            f = self.atoms.get_forces()
            e = self.atoms.get_potential_energy()
            fmax = np.sqrt((f**2).sum(axis=1).max())
            lst: list[str] = [
                f"{self.__class__.__name__}:",
                f"{f'{T[3]:02d}:{T[4]:02d}:{T[5]:02d}':>10s}",
                f"{self.nsteps:8d}",
                f"{e:10.4f}",
                f"{fmax:10.4f}",
            ]
            self.logfile.write(f"{'  '.join(lst)}\n")
            self.logfile.flush()
            self.logfile.flush()
