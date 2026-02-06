"""Generic registry pattern base class."""

from __future__ import annotations

from typing import ClassVar, Generic, TypeVar

T = TypeVar("T")


class BaseRegistry(Generic[T]):
    """Base class for component registries.

    Subclasses define _REGISTRY dict and implement register/get/available.
    Registered classes may define a ``call_name`` class attribute; if not set,
    the class name is used as the default.

    Examples:
        Create a custom registry::

            class MyComponentRegistry(BaseRegistry["MyComponent"]):
                _REGISTRY: ClassVar[dict[str, type["MyComponent"]]] = {}
                _registry_name: ClassVar[str] = "component"

            @MyComponentRegistry.register
            class ConcreteComponent:
                call_name = "concrete"

            # Look up by name
            cls = MyComponentRegistry.get("concrete")
    """

    _REGISTRY: ClassVar[dict[str, type[T]]]  # Subclass defines this
    _registry_name: ClassVar[str] = "component"  # For error messages

    @classmethod
    def register(cls, target_cls: type[T]) -> type[T]:
        """Decorator that registers a class by its ``call_name`` class attribute.

        If ``call_name`` is not defined on the class, defaults to the class name.

        Args:
            target_cls: The class to register.

        Returns:
            The registered class (unchanged).

        Raises:
            ValueError: If a component with the same name is already registered.
        """
        name = getattr(target_cls, "call_name", None) or target_cls.__name__
        if name in cls._REGISTRY:
            raise ValueError(
                f"{cls._registry_name.title()} {name!r} already registered "
                f"by {cls._REGISTRY[name].__name__}"
            )
        cls._REGISTRY[name] = target_cls
        return target_cls

    @classmethod
    def get(cls, name: str) -> type[T]:
        """Look up registered class by name.

        Args:
            name: The registered name of the component.

        Returns:
            The registered class.

        Raises:
            ValueError: If *name* is not registered.
        """
        try:
            return cls._REGISTRY[name]
        except KeyError:
            valid = ", ".join(sorted(cls._REGISTRY))
            raise ValueError(
                f"Unknown {cls._registry_name} {name!r}. Available: {valid}"
            ) from None

    @classmethod
    def available(cls) -> tuple[str, ...]:
        """Return names of all registered components.

        Returns:
            Tuple of registered component names, sorted alphabetically.
        """
        return tuple(sorted(cls._REGISTRY))
