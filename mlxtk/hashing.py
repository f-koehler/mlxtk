import subprocess


def hash_string(data: str, program: str = "sha1sum"):
    return subprocess.check_output([program], input=data.encode()).decode().split()[0]
