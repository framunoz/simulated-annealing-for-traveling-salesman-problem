"""Common interfaces for Julia wrappers."""

import abc
import typing as t

from juliacall import AnyValue  # type: ignore

__all__ = ["JuliaWrapper"]


class JuliaWrapper(abc.ABC):
    """Abstract base class for Python wrappers of Julia objects.
    
    This interface defines the contract for converting between Python and Julia objects,
    with built-in caching of the Julia representation.
    """
    _jl_obj: AnyValue | None = None

    def to_jl(self) -> AnyValue:
        """Get or create the cached Julia object.
        
        Returns:
            AnyValue: The Julia object.
        """
        if getattr(self, "_jl_obj", None) is None:
            # Use object.__setattr__ to bypass dataclass immutability if needed,
            # though here we are just caching.
            object.__setattr__(self, "_jl_obj", self._to_jl_impl())
        return self._jl_obj

    @abc.abstractmethod
    def _to_jl_impl(self) -> AnyValue:
        """Create the Julia equivalent of this Python object.
        
        This method should be implemented by subclasses to define how the
        Julia object is constructed.
        
        Returns:
            AnyValue: The Julia object.
        """
        pass

    @classmethod
    @abc.abstractmethod
    def from_jl(cls, jl_value: AnyValue) -> t.Self:
        """Create a Python object from a Julia object.
        
        Args:
            jl_value: The Julia object to convert.
            
        Returns:
            A new instance of this class wrapping the Julia object.
        """
        pass
