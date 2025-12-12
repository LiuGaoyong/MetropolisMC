from ase.calculators.calculator import Calculator
from ase.calculators.emt import EMT


def get_calculator(method: str) -> Calculator:
    if method.lower().startswith("emt"):
        return EMT()
    else:
        # TODO: add more calculators
        raise KeyError(f"Invalid SPE method: {method}")
