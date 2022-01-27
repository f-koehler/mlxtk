from mlxtk.tape import BosonicNode, NormalNode, PrimitiveNode, Node


def test_bosonic_1d():
    N = 7
    m = 13
    n = 255

    tree = BosonicNode(N)
    tree += NormalNode(m)
    tree.children[0] += PrimitiveNode(n)
    assert tree.get_tape() == [-10, N, 1, m, -1, 1, 1, 0, n, -2]

    recreated_tree = Node.from_tape(tree.get_tape())
    assert recreated_tree.get_tape() == tree.get_tape()


def test_bosonic_2d():
    N = 7
    m = 13
    nx = 255
    ny = 127

    tree = BosonicNode(N)
    normal = NormalNode(m)
    tree += normal
    normal += PrimitiveNode(nx)
    normal += PrimitiveNode(ny)

    assert tree.get_tape() == [-10, N, 1, m, -1, 1, 2, 0, nx, ny, -2]

    recreated_tree = Node.from_tape(tree.get_tape())
    assert recreated_tree.get_tape() == tree.get_tape()


def test_bose_bose_1d():
    NA = 7
    NB = 100
    MA = 5
    MB = 3
    mA = 13
    mB = 2
    n = 255
    tree = NormalNode()
    bosonic_A = BosonicNode(NA, MA)
    bosonic_B = BosonicNode(NB, MB)
    normal_A = NormalNode(mA)
    normal_B = NormalNode(mB)
    tree += bosonic_A
    tree += bosonic_B
    bosonic_A += normal_A
    bosonic_B += normal_B
    normal_A += PrimitiveNode(n)
    normal_B += PrimitiveNode(n)

    assert tree.get_tape() == [
        -10,
        2,
        0,
        MA,
        MB,
        -1,
        1,
        NA,
        1,
        mA,
        -1,
        1,
        1,
        0,
        n,
        0,
        0,
        -1,
        2,
        NB,
        1,
        mB,
        -1,
        1,
        1,
        0,
        n,
        -2,
    ]

    recreated_tree = Node.from_tape(tree.get_tape())
    assert recreated_tree.get_tape() == tree.get_tape()


def test_spin_half_binary():
    m1 = 4
    m2 = 2
    tree = NormalNode()
    for i in range(2):
        mid = NormalNode(m1)
        tree += mid
        for j in range(2):
            bottom = NormalNode(m2)
            mid += bottom
            bottom += PrimitiveNode(2)

    expectation = [
        2,
        m1,
        m1,
        -1,
        1,
        2,
        m2,
        m2,
        -1,
        1,
        1,
        2,
        0,
        -1,
        2,
        1,
        2,
        0,
        0,
        -1,
        2,
        2,
        m2,
        m2,
        -1,
        1,
        1,
        2,
        0,
        -1,
        2,
        1,
        2,
        -2,
    ]
    assert tree.get_tape() == expectation

    recreated_tree = Node.from_tape(tree.get_tape())
    assert recreated_tree.get_tape() == tree.get_tape()
