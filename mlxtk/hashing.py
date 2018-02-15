import subprocess


def hash_string(string):
    """Compute the SHA256 of a string using the ``sha256sum`` program

    Args:
        string (str): String to hash.

    Returns:
        str: SHA256 of the given string.
    """
    process = subprocess.Popen(
        ["sha256sum"], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    out, _ = process.communicate(string.encode())
    if process.wait() != 0:
        raise RuntimeError("Running \"sha256sum\" to hash string failed")
    return out.decode().split()[0]


def hash_file(path):
    """Compute the SHA256 of a file using the ``sha256sum`` program

    Args:
        path (str): Path to the file.

    Returns:
        str: SHA256 of the given file.
    """
    process = subprocess.Popen(["sha256sum", path], stdout=subprocess.PIPE)
    out, _ = process.communicate()
    if process.wait() != 0:
        raise RuntimeError("Running \"sha256sum\" to hash file failed")
    return out.decode().split()[0]


def hash_values(*args):
    return hash_string(str(len(args)) + "".join([str(arg) for arg in args]))
