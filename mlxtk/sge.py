import os
import re
import subprocess
import sys


def add_parser_arguments(parser):
    parser.add_argument(
        "--queue", default="quantix.q", help="queue of the SGE batch system")
    parser.add_argument(
        "--memory",
        default="8G",
        help="amount of memory available to the job(s)")
    parser.add_argument(
        "--time",
        default="00:10:00",
        help="maximum computation time for the job(s)")
    parser.add_argument(
        "--cpus", default="1", help="number of cpus to use for SMP")
    pass


def submit_job(jobfile):
    regex = re.compile(r"^Your job (\d+)")
    output = subprocess.check_output(["qsub", jobfile], stderr=sys.stderr)
    m = regex.match(output)

    if not m:
        raise RuntimeError("failed to extract jobid from qsub output")

    return int(m.group(1))


def mark_file_executable(path):
    os.chmod(path, 0o755)


def write_epilogue_script(path, jobid):
    script = ("#!/bin/bash\n" "qacct -j {jobid}\n").format(jobid=jobid)

    with open(path, "w") as fhandle:
        fhandle.write(script)

    mark_file_executable(path)


def write_job_file(path, name, cmd, args, output=None):
    script = [
        # yapf: disable
        "#!/bin/bash",
        "#$ -N {name}",
        "#$ -q {queue}",
        "#$ -S /bin/bash",
        "#$ -cwd",
        "#$ -j y",
        "#$ -V",
        "#$ -l h_vmem={memory}",
        "#$ -l h_cpu={time}",
        "#$ -pe smp {cpus}"
        # yapf: enable
    ]
    if output:
        script.append("#$ -o " + output)
    script.append("{cmd}")

    script = "\n".join(script).format(
        name=name,
        queue=args.queue,
        memory=args.memory,
        time=args.time,
        cpus=args.cpus,
        cmd=cmd,
        output=output)

    with open(path, "w") as fhandle:
        fhandle.write(script)

    mark_file_executable(path)


def write_stop_script(path, jobids):
    script = ["#!/bin/bash", "qdel {jobids}"]
    script = "\n".join(script).format(jobids=" ".join(
        [str(jobid) for jobid in jobids]))

    with open(path, "w") as fhandle:
        fhandle.write(script)

    mark_file_executable(path)
