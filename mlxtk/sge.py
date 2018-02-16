import os
import re
import subprocess
import sys

from mlxtk import log

logger = log.get_logger(__name__)


def add_parser_arguments(parser):
    """Add SGE related command line options to an :py:class:`argparse.ArgumentParser`

    Args:
        parser (argparse.ArgumentParser): parser to modify
    """
    parser.add_argument(
        "--queue",
        default="none",
        help=
        "queue of the SGE batch system, \"none\" if you do not want to specify a queue"
    )
    parser.add_argument(
        "--memory",
        default="2G",
        help="amount of memory available to the job(s)")
    parser.add_argument(
        "--time",
        default="00:10:00",
        help="maximum computation time for the job(s)")
    parser.add_argument(
        "--cpus", default="1", help="number of cpus to use for SMP")
    parser.add_argument(
        "--email",
        default="none",
        help=
        "email address to notify about finished, aborted and suspended jobs")


def submit_job(jobfile):
    """Submit a job to the SGE batch-queuing system

    Args:
        jobfile (str): Path to the script that runs the job

    Returns:
        int: Id of the newly created job
    """
    logger.info("submit job script \"%s\"", jobfile)
    regex = re.compile(r"^Your job (\d+)")
    output = subprocess.check_output(["qsub", jobfile])
    m = regex.match(output.decode())

    if not m:
        raise RuntimeError("failed to extract jobid from qsub output")

    return int(m.group(1))


def mark_file_executable(path):
    """Make a file executable using :py:func:`os.chmod`

    Args:
        path (str): Path to the file
    """
    logger.info("make file executable \"%s\"", path)
    os.chmod(path, 0o755)


def write_epilogue_script(path, jobid):
    """Create a epilogue script fo a given job

    The script uses the `qacct` command to show information about the job

    Args:
        path (str): Path where the script should be stored
        jobid (int): Id of the job
    """
    script = ("#!/bin/bash\n" "qacct -j {jobid}\n").format(jobid=jobid)

    logger.info("write epilogue script \"%s\"", path)
    with open(path, "w") as fhandle:
        fhandle.write(script)

    mark_file_executable(path)


def write_job_file(path, name, cmd, args):
    """Create a job file

    Args:
        path (str): Path where the scirpt should be stored
        name (str): Name of the job
        cmd (str): Job command
        args (argparse.Namespace): Namespace containing the SGE related command line arguments
    """
    script = ["#!/bin/bash", "#$ -N {name}"]
    if args.queue.upper() != "NONE":
        script.append("#$ -q {queue}")
    if args.email.upper() != "NONE":
        script.append("#$ -M {email} -m aes")
    script += [
        "#$ -S /bin/bash", "#$ -cwd", "#$ -j y",
        "#$ -V", "#$ -l h_vmem={memory}", "#$ -l h_cpu={time}",
        "#$ -pe smp {cpus}", "export OMP_NUM_THREADS={cpus}", "{cmd}\n"
    ]

    script = "\n".join(script).format(
        name=name,
        queue=args.queue,
        memory=args.memory,
        time=args.time,
        cpus=args.cpus,
        cmd=cmd,
        email=args.email)

    logger.info("write job script \"%s\"", path)
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

    logger.info("write stop script \"%s\"", path)
    with open(path, "w") as fhandle:
        fhandle.write(script)

    mark_file_executable(path)
