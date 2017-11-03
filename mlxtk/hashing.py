import subprocess


def hash_string(string):
    process = subprocess.Popen(
        ["sha256sum"], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    out, _ = process.communicate(string.encode())
    if process.wait() != 0:
        raise RuntimeError("Running \"sha256sum\" to hash string failed")
    return out.decode().split()[0]


def hash_file(path):
    process = subprocess.Popen(["sha256sum", path], stdout=subprocess.PIPE)
    out, _ = process.communicate()
    if process.wait() != 0:
        raise RuntimeError("Running \"sha256sum\" to hash file failed")
    return out.decode().split()[0]


def hash_values(*args):
    return hash_string(str(len(args)) + "".join([str(arg) for arg in args]))
