"""
Abstract Base Classes for all FEA model definitions.
"""
from abc import ABC, abstractmethod

class Material(ABC):
    """Abstract base class for a material model."""
    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def get_symbolic_model(self):
        """
        Returns the symbolic representation of the material.

        Returns:
            A tuple containing:
            - params (list): List of SymPy symbols for material parameters (e.g., [E, nu]).
            - D_matrix (sympy.Matrix): The 6x6 symbolic constitutive matrix.
        """
        pass


class Element(ABC):
    """Abstract base class for a finite element."""
    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def get_symbolic_model(self, material: Material):
        """
        Returns the full mathematical model for the element combined with a material.

        Args:
            material (Material): A material object providing the constitutive model.

        Returns:
            MathModel: A data container with symbolic inputs and outputs for the code generator.
        """
        pass
