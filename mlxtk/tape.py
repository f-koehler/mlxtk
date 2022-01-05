from __future__ import annotations

import abc
from typing import Any, Generator

from graphviz import Digraph


class Node(abc.ABC):
    def __init__(self, dimension: int, parent: Node | None = None):
        self.dimension = dimension
        self.parent = parent
        self.children: list[Node] = []
        self.has_indist_node = False
        self.attrs: dict[str, Any] = {}

    def __iadd__(self, other: Node) -> Node:
        other.parent = self
        self.children.append(other)

        if other.is_indistinguishable():
            self.get_root().has_indist_node = True

        return self

    def compute_node_indices(self, counter: int | None = None) -> int:
        if counter is None:
            if self.is_root():
                counter = 0
            else:
                self.get_root().compute_node_indices(counter)
                return

        self.attrs["index"] = counter
        counter += 1
        for child in self.children:
            counter = child.compute_node_indices(counter)
        return counter

    def compute_dof_numbers(self, counter: int = 0) -> int:
        if counter is None:
            if self.is_root():
                counter = 0
            else:
                self.get_root().compute_dof_numbers(counter)
                return

        if self.is_primitive():
            self.attrs["dof"] = counter
            counter += 1
        else:
            for child in self.children:
                counter = child.compute_dof_numbers(counter)

        return counter

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

    def get_primitive_nodes(self) -> list[PrimitiveNode]:
        if self.is_primitive():
            return [self]

        result: list[PrimitiveNode] = []
        for child in self.children:
            result += child.get_primitive_nodes()
        return result

    def to_graph(self):
        self.compute_node_indices()
        self.compute_dof_numbers()

        g = Digraph()
        g.attr(rankdir="TB", nodesep="1.5", ranksep="1.0")
        self.add_to_graph(g)
        return g

    def add_to_graph(self, g: Digraph, parent: str | None = None):
        if isinstance(self, PrimitiveNode):
            shape = "square"
            dof = self.attrs["dof"]
            xlabel = f"{dof=}"
        elif isinstance(self, BosonicNode) or isinstance(self, FermionicNode):
            shape = "doublecircle"
            N = self.particles
            xlabel = "{N=}"
        else:
            shape = "circle"
            xlabel = None

        for attr in self.attrs:
            if attr == "dof":
                continue
            if attr == "index":
                continue
            if xlabel is None:
                xlabel = ""
            xlabel += f"\n{attr}={self.attrs[attr]}"

        name = str(self.attrs["index"] + 1)
        g.node(name, shape=shape, xlabel=xlabel)

        if parent is not None:
            g.edge(parent, name, label=f"{self.dimension}")

        for child in self.children:
            child.add_to_graph(g, name)


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
