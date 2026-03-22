"""Registry for named transform functions."""

from typing import Callable


class TransformRegistry:
    """Registry for named transform functions.

    Each transform is a callable that takes an Ibis table expression
    and returns a new Ibis table expression with additional/modified columns.
    """

    def __init__(self):
        self._transforms: dict[str, Callable] = {}

    def register(self, name: str, fn: Callable) -> None:
        self._transforms[name] = fn

    def get(self, name: str) -> Callable:
        return self._transforms[name]

    def __contains__(self, name: str) -> bool:
        return name in self._transforms

    def list_transforms(self) -> list[str]:
        return list(self._transforms.keys())
