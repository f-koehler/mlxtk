import mlxtk.hash

import os.path
import sys

if sys.version_info <= (3, 0):
    from StringIO import StringIO
else:
    from io import StringIO

from QDTK.Operator import OCoef as Coef
from QDTK.Operator import OTerm as Term
from QDTK.Operator import Operator

from QDTK.Operatorb import OCoef as Coefb
from QDTK.Operatorb import OTerm as Termb
from QDTK.Operatorb import Operatorb


def write_operator(operator, path):
    """Write a :class:`QDTK.Operator.Operator`/:class:`QDTK.Operatorb.Operatorb` to a file.

    The operator is written to the specified path. If the file is already
    existent and its hash is compared to the given operator. A write takes only
    place if the file contains another operator.

    Args:
        operator: operator to write
        path: path that the operator should be written to

    Returns:
        bool: ``True`` if the file was created/updated, ``False`` if the file was already up to date.
    """
    f = StringIO()
    if isinstance(operator, Operator):
        operator.createOperatorFile(f)
    elif isinstance(operator, Operatorb):
        operator.createOperatorFileb(f)

    s = f.getvalue()

    if os.path.exists(path):
        h_new = mlxtk.hash.hash_string(s)
        h_old = mlxtk.hash.hash_file(path)

        if h_new == h_old:
            return False

    with open(path, "w") as fh:
        fh.write(s)

    return True
