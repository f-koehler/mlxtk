import subprocess

hash_program = "sha512sum"
"""str: name of the program to be used to compute hashes

This program is expected to read its data from stdin and write the output to
standard output. The output should be formatted as in the common unix hash
utilities (like ``sha512sum``). The first column should be the hash value
followed by the filename.
"""


def hash_string(string):
    """Compute the hash of a :class:`str`.

    The given string is piped into the hash program specified by
    :data:`mlxtk.hash.hash_string`. The output is split at the first whitespace
    and the first entry is used as the hash. This works fine with default unix
    tools like ``sha512sum``.

    Args:
        string (str): string of which the hash is to be computed

    Returns:
        str: hash value as a string
    """
    process = subprocess.Popen(
        ["sha512sum"], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    out, _ = process.communicate(string.encode())
    if process.returncode:
        raise subprocess.CalledProcessError()
    return out.decode().split()[0].strip()


def hash_dict(d):
    """Compute the hash of a :class:`dict`.

    The dictionary is converted to a string and passed to
    :func:`mlxtk.hash.hash_string`.

    Args:
        d (dict): dictionary of which the hash is to be computed

    Returns:
        str: hash value as a string
    """
    entries = [str(len(d.keys()))]
    entries += [str(key) + str(d[key]) for key in d]
    return hash_string("".join(entries))


def hash_list(l):
    """Compute the hash of a :class:`list`.

    The list is converted to a string and passed to
    :func:`mlxtk.hash.hash_string`.

    Args:
        l (list): list of which the hash is to be computed

    Returns:
        str: hash value as a string
    """
    entries = [str(len(l))]
    entries += [str(i) + str(l[i]) for i in range(0, len(l))]
    return hash_string("".join(entries))


def hash_file(path):
    """Compute the hash of a file.

    The file is read completely (at once) and passed to
    :func:`mlxtk.hash.hash_string`.

    Args:
        path (str): path of the file which is to be hashed

    Returns:
        str: hash value as a string
    """
    with open(path) as f:
        return hash_string(f.read())
