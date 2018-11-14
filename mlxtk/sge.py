"""Work with the SGE scheduling system.
"""
import argparse
import os
import re
import subprocess
from typing import List

from . import log, templates

LOGGER = log.get_logger(__name__)
REGEX_QSTAT = re.compile(r"^(\d+)\s+")
REGEX_QSUB = re.compile(r"^Your job (\d+)")


def add_parser_arguments(parser: argparse.ArgumentParser):
    """Add SGE related command line options to an
    :py:class:`argparse.ArgumentParser`.

    Args:
        parser (argparse.ArgumentParser): parser to modify
    """
    parser.add_argument(
        "--queues",
        default="none",
        help=("comma separated list of queues for the SGE batch system,"
              ' "none" if you do not want to specify a queue'),
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
        help=("email address to notify about finished, aborted and suspended"
              "jobs"),
    )


def get_jobs_in_queue() -> List[int]:
    """Get the job ids of all jobs in the SGE queue.

    Returns:
        List[int]: job ids of running jobs
    """
    output = subprocess.check_output(["qstat"]).decode().splitlines()
    job_ids = []
    for line in output:
        m = REGEX_QSTAT.match(line)
        if m:
            job_ids.append(int(m.group(1)))
    return job_ids


def submit(command: str, args, sge_dir: str = os.path.curdir,
           job_name="") -> int:
    """Create a jobfile for a command and submit it.

    This function takes an arbitrary shell command and submits it to SGE.
    Therefore the following files are created:
    - ``sge.id``: a file containing the job id
    - ``sge_job``: the job script that is submitted using ``qsub``
    - ``sge_stop``: a script to stop this job
    - ``sge_epilogue``: a script to gather accounting information using
      ``qacct``.

   Before doing anything this method checks if the ``sge.id`` file exists and
   checks whether the corresponding job is still running.

    Args:
        command (str): the shell command as a string.
        args (argparse.Namespace): the command line arguments
        sge_dir (str): working dir for the job

    Returns:
        int: job id of the new job
    """
    id_file = os.path.join(sge_dir, "sge.id")
    job_script = os.path.join(sge_dir, "sge_job")
    stop_script = os.path.join(sge_dir, "sge_stop")
    epilogue_script = os.path.join(sge_dir, "sge_epilogue")

    # check if job is already running
    if os.path.exists(id_file):
        with open(id_file) as fp:
            job_id = int(fp.read())
            if job_id in get_jobs_in_queue():
                LOGGER.error(
                    "job seems to be in the queue already (with id %d)",
                    job_id)
                return -1

    # set up args for the job script
    args = {
        "command": command,
        "cpus": args.cpus,
        "email": args.email,
        "memory": args.memory,
        "queues": args.queues,
        "sge_dir": sge_dir,
        "time": args.time,
    }
    if job_name:
        args["job_name"] = job_name

    # create job script
    LOGGER.debug("create job script")
    with open(job_script, "w") as fp:
        fp.write(templates.get_template("sge_job.j2").render(args=args))
    os.chmod(job_script, 0o755)
    LOGGER.debug("done")

    # submit job
    LOGGER.debug("submit job")
    output = subprocess.check_output(["qsub", job_script]).decode()
    m = REGEX_QSUB.match(output)
    job_id = int(m.group(1))
    LOGGER.debug("done")

    LOGGER.debug("write id to file")
    with open(id_file, "w") as fp:
        fp.write(str(job_id) + "\n")
    LOGGER.debug("done")

    # write stop script
    LOGGER.debug("create stop script")
    with open(stop_script, "w") as fp:
        fp.write(templates.get_template("sge_stop.j2").render(job_id=job_id))
    os.chmod(stop_script, 0o755)
    LOGGER.debug("done")

    # write epilogue script
    LOGGER.debug("create epilogue script")
    with open(epilogue_script, "w") as fp:
        fp.write(
            templates.get_template("sge_epilogue.j2").render(job_id=job_id))
    os.chmod(epilogue_script, 0o755)
    LOGGER.debug("done")

    return job_id
