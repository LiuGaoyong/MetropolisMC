from ._base import MonteCarloState as State


class CanonicalState(State):
    """The Monte Carlo state for the canonical ensemble."""

    _state_keys: set[str] = {"U"}


class GrandCanonicalState(State):
    """The Monte Carlo state for the grand canonical ensemble."""

    _state_keys: set[str] = {"U", "N"}


class IsothermalIsobaricState(State):
    """The Monte Carlo state for the Isothermal–isobaric ensemble."""

    _state_keys: set[str] = {"U", "V"}


class MicroCanonicalState(State):
    """The Monte Carlo state for the micro-canonical ensemble."""

    _state_keys: set[str] = {"K"}
