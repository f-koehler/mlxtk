import subprocess


def get_terminal_size():
    tmp = subprocess.check_output(["stty", "size"]).decode().split()
    return int(tmp[0]), int(tmp[1])
