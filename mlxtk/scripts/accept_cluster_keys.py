import re
import shutil
import subprocess

from mlxtk import rsync

regex_hostname = re.compile(r"^(.+)\.physnet\.uni-hamburg\.de$")


def main():
    if not shutil.which("sshpass"):
        raise RuntimeError("Cannot find sshpass executable")

    if not shutil.which("qconf"):
        raise RuntimeError("Cannot find qconf executable")

    lines = subprocess.check_output(["qconf", "-sel"]).decode().splitlines()
    nodes = []
    for line in lines:
        m = regex_hostname.match(line)
        if m:
            nodes.append(m.group(1))

    with rsync.SSHPassword() as password:
        for node in nodes:
            print("node:", node)
            cmd = [
                "sshpass", "-p", password, "ssh", "-oStrictHostKeyChecking=no",
                node, "uptime"
            ]
            subprocess.check_output(cmd)


if __name__ == "__main__":
    main()
