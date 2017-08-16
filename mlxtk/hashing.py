import hashlib


def hash_file(path):
    hfunc = hashlib.sha256()
    with open(path, "rb") as fhandle:
        hfunc.update(fhandle.read())
    return hfunc.hexdigest()


def hash_string(string):
    hfunc = hashlib.sha256()
    hfunc.update(string.encode())
    return hfunc.hexdigest()


def hash_values(*args):
    hfunc = hashlib.sha256()
    string = str(len(args)) + "".join([str(arg) for arg in args])
    hfunc.update(string.encode())
    return hfunc.hexdigest()
