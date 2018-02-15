import os
import shutil

OVERRIDES = {}
"""dict: A dictionary containing overridden qdtk_executables"""


def find_qdtk_executable(name):
    if name in OVERRIDES:
        return OVERRIDES[name]

    if "QDTK_PREFIX" in os.environ:
        path = os.path.join(os.environ["QDTK_PREFIX"], "bin", name)
        if os.path.isfile(path) and os.access(path, os.X_OK):
            return path
        else:
            raise RuntimeError(
                "Cannot find qdtk_executable \"" + name + "\" in prefix \"" +
                os.environ["QDTK_PREFIX"] + "\"")

    path = shutil.which(name)
    if not path:
        raise RuntimeError("Cannot find qdtk_executable \"" + name + "\"")
    return path


def override(name, path):
    """Override a qdtk_executable

    Args:
        name (str): Name of the executable to override, e.g. `qdtk_propagate.x`
        path (str): Path to the executable to use instead of the default one.
    """
    OVERRIDES[name] = path
