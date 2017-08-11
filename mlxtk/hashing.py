import hashlib


def hash_file(file):
    h = hashlib.sha256()
    with open(file, "rb") as fh:
        h.update(fh.read())
        return h.hexdigest()


def hash_string(string):
    h = hashlib.sha256()
    h.update(string.encode())
    return h.hexdigest()


def hash_values(*args):
    h = hashlib.sha256()
    string = str(len(args)) + "".join([str(arg) for arg in args])
    h.update(string.encode())
    return h.hexdigest()
