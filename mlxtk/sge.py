import os
import re
import subprocess
import sys


def add_parser_arguments(parser):
    """Add SGE related command line options to an :py:class:`argparse.ArgumentParser`

    Args:
        parser (argparse.ArgumentParser): parser to modify
    """
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


def submit_job(jobfile):
    """Submit a job to the SGE batch-queuing system

    Args:
        jobfile (str): Path to the script that runs the job

    Returns:
        int: Id of the newly created job
    """
    regex = re.compile(r"^Your job (\d+)")
    output = subprocess.check_output(["qsub", jobfile], stderr=sys.stderr)
    m = regex.match(output)

    if not m:
        raise RuntimeError("failed to extract jobid from qsub output")

    return int(m.group(1))


def mark_file_executable(path):
    """Make a file executable using :py:func:`os.chmod`

    Args:
        path (str): Path to the file
    """
    os.chmod(path, 0o755)


def write_epilogue_script(path, jobid):
    """Create a epilogue script fo a given job

    The script uses the `qacct` command to show information about the job

    Args:
        path (str): Path where the script should be stored
        jobid (int): Id of the job
    """
    script = ("#!/bin/bash\n" "qacct -j {jobid}\n").format(jobid=jobid)

    with open(path, "w") as fhandle:
        fhandle.write(script)

    mark_file_executable(path)


def write_job_file(path, name, cmd, args, output=None):
    """Create a job file

    Args:
        path (str): Path where the scirpt should be stored
        name (str): Name of the job
        cmd (str): Job command
        args (argparse.Namespace): Namespace containing the SGE related command line arguments
        output (str): Path where the stdout should be written, `None` if not desired
    """
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
    """Create a stop script to abort given jobs

    Args:
        path (str): Path where the script should be stored
        jobids (list): List of job ids to abort
    """
    script = ["#!/bin/bash", "qdel {jobids}"]
    script = "\n".join(script).format(jobids=" ".join(
        [str(jobid) for jobid in jobids]))

    with open(path, "w") as fhandle:
        fhandle.write(script)

    mark_file_executable(path)
