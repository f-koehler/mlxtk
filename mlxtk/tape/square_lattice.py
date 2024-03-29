from __future__ import annotations

import itertools
from dataclasses import dataclass
from multiprocessing.sharedctypes import Value

from .nodes import Node, NormalNode, PrimitiveNode


@dataclass
class QuadTreeLayer:
    size: int
    nodes: dict[tuple[int, int], Node]

    @staticmethod
    def create_bottom(L: int, primitive_dim: int) -> QuadTreeLayer:
        if (not L) or (L & (L - 1)):
            raise ValueError("L must be a positive power of two")

        return QuadTreeLayer(
            L,
            {
                (x, y): PrimitiveNode(primitive_dim, attrs={"x": x, "y": y})
                for x, y in itertools.product(range(L), range(L))
            },
        )

    def create_parent(self, orbitals: int = 1) -> QuadTreeLayer:
        if self.size == 1:
            raise ValueError("Already reached top layer, cannot create parent layer")

        new_size = self.size // 2

        layer = QuadTreeLayer(
            new_size,
            {
                (x, y): NormalNode(orbitals, attrs={"x": x, "y": y})
                for x, y in itertools.product(range(new_size), range(new_size))
            },
        )

        for x, y in itertools.product(range(new_size), range(new_size)):
            layer.nodes[(x, y)] += self.nodes[(2 * x, 2 * y)]
            layer.nodes[(x, y)] += self.nodes[(2 * x + 1, 2 * y)]
            layer.nodes[(x, y)] += self.nodes[(2 * x, 2 * y + 1)]
            layer.nodes[(x, y)] += self.nodes[(2 * x + 1, 2 * y + 1)]

        return layer


def create_quad_tree(
    L: int,
    primitive_dim: int,
    orbitals: list[int],
) -> tuple[NormalNode, dict[tuple(int, int), int]]:
    orbitals_ = orbitals.copy()
    orbitals_.append(1)
    top_layer = QuadTreeLayer.create_bottom(L, primitive_dim)
    while top_layer.size != 1:
        top_layer = top_layer.create_parent(orbitals_.pop(0))

    if orbitals_:
        raise ValueError(f"Too many orbital numbers provided, remainder: {orbitals_}")

    top_node = top_layer.nodes[(0, 0)]
    top_node.compute_layers()
    top_node.compute_dof_numbers()

    dof_map: dict[tuple(int, int), int] = {}
    for node in top_node.get_primitive_nodes():
        dof_map[(node.attrs["x"], node.attrs["y"])] = node.attrs["dof"]

    return top_node, dof_map


@dataclass
class AlternatingBinaryTreeLayer:
    even: bool
    size_x: int
    size_y: int
    nodes: dict[tuple[int, int], Node]

    @staticmethod
    def create_bottom(L: int, primitive_dim: int) -> AlternatingBinaryTreeLayer:
        if (not L) or (L & (L - 1)):
            raise ValueError("L must be a positive power of two")

        return AlternatingBinaryTreeLayer(
            True,
            L,
            L,
            {
                (x, y): PrimitiveNode(primitive_dim, attrs={"x": x, "y": y})
                for x, y in itertools.product(range(L), range(L))
            },
        )

    def create_parent(self, orbitals: int = 1):
        if (self.size_x == 1) and (self.size_y == 1):
            raise ValueError("Already reached top layer, cannot create parent layer")

        if self.even:
            new_size_x = self.size_x // 2
            new_size_y = self.size_y
            layer = AlternatingBinaryTreeLayer(
                False,
                new_size_x,
                new_size_y,
                {
                    (x, y): NormalNode(orbitals, attrs={"x": x, "y": y})
                    for x, y in itertools.product(range(new_size_x), range(new_size_y))
                },
            )
            for x, y in itertools.product(range(new_size_x), range(new_size_y)):
                layer.nodes[(x, y)] += self.nodes[(2 * x, y)]
                layer.nodes[(x, y)] += self.nodes[(2 * x + 1, y)]
        else:
            new_size_x = self.size_x
            new_size_y = self.size_y // 2
            layer = AlternatingBinaryTreeLayer(
                True,
                new_size_x,
                new_size_y,
                {
                    (x, y): NormalNode(orbitals, attrs={"x": x, "y": y})
                    for x, y in itertools.product(range(new_size_x), range(new_size_y))
                },
            )
            for x, y in itertools.product(range(new_size_x), range(new_size_y)):
                layer.nodes[(x, y)] += self.nodes[(x, 2 * y)]
                layer.nodes[(x, y)] += self.nodes[(x, 2 * y + 1)]

        return layer


def create_alternating_binary_tree(
    L: int,
    primitive_dim: int,
    orbitals: list[int],
) -> tuple[NormalNode, dict[tuple[int, int], int]]:
    orbitals_ = orbitals.copy()
    orbitals_.append(1)
    top_layer = AlternatingBinaryTreeLayer.create_bottom(L, primitive_dim)
    while (top_layer.size_x > 1) or (top_layer.size_y > 1):
        top_layer = top_layer.create_parent(orbitals_.pop(0))

    if orbitals_:
        raise ValueError(f"Too many orbital numbers provided, remainder: {orbitals_}")

    top_node = top_layer.nodes[(0, 0)]
    top_node.compute_layers()
    top_node.compute_dof_numbers()

    dof_map: dict[tuple[int, int], int] = {}
    for node in top_node.get_primitive_nodes():
        dof_map[(node.attrs["x"], node.attrs["y"])] = node.attrs["dof"]

    return top_node, dof_map


@dataclass
class AlternatingBinaryQuadLowestTreeLayer:
    even: bool
    size_x: int
    size_y: int
    nodes: dict[tuple[int, int], Node]
    height: int

    @staticmethod
    def create_bottom(
        L: int,
        primitive_dim: int,
    ) -> AlternatingBinaryQuadLowestTreeLayer:
        if (not L) or (L & (L - 1)):
            raise ValueError("L must be a positive power of two")

        return AlternatingBinaryQuadLowestTreeLayer(
            True,
            L,
            L,
            {
                (x, y): PrimitiveNode(primitive_dim, attrs={"x": x, "y": y})
                for x, y in itertools.product(range(L), range(L))
            },
            0,
        )

    def create_parent(self, orbitals: int = 1):
        if (self.size_x == 1) and (self.size_y == 1):
            raise ValueError("Already reached top layer, cannot create parent layer")

        if self.even:
            if self.height == 0:
                new_size_x = self.size_x // 2
                new_size_y = self.size_y // 2
                layer = AlternatingBinaryQuadLowestTreeLayer(
                    False,
                    new_size_x,
                    new_size_y,
                    {
                        (x, y): NormalNode(orbitals, attrs={"x": x, "y": y})
                        for x, y in itertools.product(
                            range(new_size_x),
                            range(new_size_y),
                        )
                    },
                    self.height + 1,
                )
                for x, y in itertools.product(range(new_size_x), range(new_size_y)):
                    layer.nodes[(x, y)] += self.nodes[(2 * x, 2 * y)]
                    layer.nodes[(x, y)] += self.nodes[(2 * x + 1, 2 * y)]
                    layer.nodes[(x, y)] += self.nodes[(2 * x, 2 * y + 1)]
                    layer.nodes[(x, y)] += self.nodes[(2 * x + 1, 2 * y + 1)]
            else:
                new_size_x = self.size_x // 2
                new_size_y = self.size_y
                layer = AlternatingBinaryQuadLowestTreeLayer(
                    False,
                    new_size_x,
                    new_size_y,
                    {
                        (x, y): NormalNode(orbitals, attrs={"x": x, "y": y})
                        for x, y in itertools.product(
                            range(new_size_x),
                            range(new_size_y),
                        )
                    },
                    self.height + 1,
                )
                for x, y in itertools.product(range(new_size_x), range(new_size_y)):
                    layer.nodes[(x, y)] += self.nodes[(2 * x, y)]
                    layer.nodes[(x, y)] += self.nodes[(2 * x + 1, y)]
        else:
            new_size_x = self.size_x
            new_size_y = self.size_y // 2
            layer = AlternatingBinaryQuadLowestTreeLayer(
                True,
                new_size_x,
                new_size_y,
                {
                    (x, y): NormalNode(orbitals, attrs={"x": x, "y": y})
                    for x, y in itertools.product(range(new_size_x), range(new_size_y))
                },
                self.height + 1,
            )
            for x, y in itertools.product(range(new_size_x), range(new_size_y)):
                layer.nodes[(x, y)] += self.nodes[(x, 2 * y)]
                layer.nodes[(x, y)] += self.nodes[(x, 2 * y + 1)]

        return layer


def create_alteranting_binary_quad_lowest_tree(
    L: int,
    primitive_dim: int,
    orbitals: list[int],
) -> tuple[NormalNode, dict[tuple[int, int], int]]:
    orbitals_ = orbitals.copy()
    orbitals_.append(1)
    top_layer = AlternatingBinaryQuadLowestTreeLayer.create_bottom(L, primitive_dim)
    while (top_layer.size_x > 1) and (top_layer.size_y > 1):
        top_layer = top_layer.create_parent(orbitals_.pop(0))

    if orbitals_:
        raise ValueError(f"Too many orbital numbers provided, remainder: {orbitals_}")

    top_node = top_layer.nodes[(0, 0)]
    top_node.compute_layers()
    top_node.compute_dof_numbers()

    dof_map: dict[tuple[int, int], int] = {}
    for node in top_node.get_primitive_nodes():
        dof_map[(node.attrs["x"], node.attrs["y"])] = node.attrs["dof"]

    return top_node, dof_map
