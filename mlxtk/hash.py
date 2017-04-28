import hashlib

hash_algorithm = hashlib.sha512


def hash_dict(d):
    entries = [str(len(d.keys()))]
    entries += [str(key) + str(d[key]) for key in d]
    bytes = ("".join(entries)).encode()
    return hash_algorithm(bytes).hexdigest()


def hash_list(l):
    entries = [str(len(l))]
    entries += [str(i) + str(l[i]) for i in range(0, len(l))]
    bytes = ("".join(entries)).encode()
    return hash_algorithm(bytes).hexdigest()
