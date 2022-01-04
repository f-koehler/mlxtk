from __future__ import annotations
import abc
from typing import Type


class Node(abc.ABC):
    def __init__(self, dimension: int, parent: Node | None = None):
        self.dimension = dimension
        self.parent = parent
        self.children: list[Node] = []
        self.has_indist_node = False

    def __iadd__(self, other: Node) -> Node:
        other.parent = self
        self.children.append(other)

        if other.is_indistinguishable():
            self.get_root().has_indist_node = True

        return self

    def is_root(self) -> bool:
        return self.parent is None

    def is_primitive(self) -> bool:
        return isinstance(self, PrimitiveNode)

    def is_indistinguishable(self) -> bool:
        return isinstance(self, BosonicNode) or isinstance(self, FermionicNode)

    def has_indistinguishable_node(self) -> bool:
        return self.get_root().has_indist_node

    def get_root(self) -> Node:
        if self.is_root():
            return self

        return self.parent.get_root()

    @abc.abstractmethod
    def get_node_tape(self) -> list[int]:
        return []

    def get_tape(self) -> list[int]:
        tape: list[int] = []

        if self.is_root() and self.has_indistinguishable_node():
            tape.append(-10)

        tape += self.get_node_tape()
        for i, child in enumerate(self.children):
            if child.is_primitive():
                continue

            tape += [-1, i + 1]
            tape += child.get_tape()
            tape.append(0)

        if self.is_root():
            while tape[-1] == 0:
                del tape[-1]
            tape.append(-2)
        return tape


class NormalNode(Node):
    def __init__(self, orbitals: int = 1, parent: Node | None = None):
        super().__init__(orbitals, parent)
        self.orbitals = orbitals

    def get_node_tape(self) -> list[int]:
        tape = [
            len(self.children),
        ]
        if self.has_indistinguishable_node():
            tape.append(0)
        for child in self.children:
            tape.append(child.dimension)
        return tape


class BosonicNode(Node):
    def __init__(self, particles: int, orbitals: int = 1, parent: Node | None = None):
        super().__init__(orbitals, parent)
        self.orbitals = orbitals
        self.particles = particles
        self.has_indist_node = True

    def get_node_tape(self) -> list[int]:
        if len(self.children) != 1:
            raise ValueError("expected exactly one child for bosonic node")

        child = self.children[0]

        if not isinstance(child, NormalNode):
            raise TypeError("expected normal node as child for bosonic node")

        return [self.particles, +1, child.dimension]


class FermionicNode(Node):
    def __init__(self, particles: int, orbitals: int, parent: Node | None = None):
        super().__init__(orbitals, parent)

        if orbitals < particles:
            raise ValueError("expected at least one orbital per fermion")

        self.orbitals = orbitals
        self.particles = particles
        self.has_indist_node = True

    def get_node_tape(self) -> list[int]:
        if len(self.children) != 1:
            raise ValueError("expected exactly one child for fermionic node")
        return [self.particles, -1, self.orbitals]


class PrimitiveNode(Node):
    def __init__(self, grid_points: int, parent: Node | None = None) -> PrimitiveNode:
        super().__init__(grid_points, parent)
        self.grid_points = grid_points

    def get_node_tape(self) -> list[int]:
        return []

    def __iadd__(self, other: Node) -> Node:
        raise TypeError("cannot add child nodes to primitive node")
