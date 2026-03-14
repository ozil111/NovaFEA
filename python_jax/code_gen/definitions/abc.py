"""
Abstract Base Classes for all FEA model definitions.
"""
from abc import ABC, abstractmethod

class Material(ABC):
    """Abstract base class for a material model."""
    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def get_constitutive_model(self):
        """
        Returns the symbolic representation of the material's constitutive model (D-matrix).

        Returns:
            MathModel: A data container for generating the D-matrix from material parameters.
        """
        pass


class Element(ABC):
    """Abstract base class for a finite element."""
    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def get_stiffness_model(self):
        """
        Returns the mathematical model for the element's stiffness kernel.
        This model takes nodal coordinates and D-matrix components as symbolic inputs.

        Returns:
            MathModel: A data container for generating the Ke matrix.
        """
        pass
