import os
import shutil

OVERRIDES = {}


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
    OVERRIDES[name] = path
