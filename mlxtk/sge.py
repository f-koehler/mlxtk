import os
import re
import subprocess

from . import log, templates

LOGGER = log.get_logger(__name__)
REGEX_QSTAT = re.compile(r"^(\d+)\s+")
REGEX_QSUB = re.compile(r"^Your job (\d+)")


def add_parser_arguments(parser):
    """Add SGE related command line options to an :py:class:`argparse.ArgumentParser`

    Args:
        parser (argparse.ArgumentParser): parser to modify
    """
    parser.add_argument(
        "--queues",
        default="none",
        help='comma separated list of queues for the SGE batch system, "none" if you do not want to specify a queue',
    )
    parser.add_argument(
        "--memory", default="2G", help="amount of memory available to the job(s)"
    )
    parser.add_argument(
        "--time", default="00:10:00", help="maximum computation time for the job(s)"
    )
    parser.add_argument("--cpus", default="1", help="number of cpus to use for SMP")
    parser.add_argument(
        "--email",
        default="none",
        help="email address to notify about finished, aborted and suspended jobs",
    )


def get_jobs_in_queue():
    output = subprocess.check_output(["qstat"]).decode().splitlines()
    job_ids = []
    for line in output:
        m = REGEX_QSTAT
        if m:
            job_ids.append(int(m.group(1)))
    return job_ids


def submit(command, args):
    # check if job is already running
    if os.path.exists("sge.id"):
        with open("sge.id") as fp:
            job_id = int(fp.read())
            if job_id in get_jobs_in_queue():
                LOGGER.warning(
                    "job seems to be in the queue already (with id %d)", job_id
                )
                return

    # set up args for the job script
    args = {
        "queues": args.queues,
        "memory": args.memory,
        "time": args.time,
        "cpus": args.cpus,
        "email": args.email,
    }

    # create job script
    LOGGER.debug("create job script")
    with open("sge", "w") as fp:
        fp.write(templates.get_template("sge_job.j2").render(args=args))
    os.chmod("sge", 0o755)
    LOGGER.debug("done")

    # submit job
    LOGGER.debug("submit job")
    output = subprocess.check_output(["qsub", "sge"]).decode()
    m = REGEX_QSUB.match(output)
    job_id = int(m.group(1))
    LOGGER.debug("done")

    LOGGER.debug("write id to file")
    with open("sge.id", "w") as fp:
        fp.write(str(job_id) + "\n")
    LOGGER.debug("done")

    # write stop script
    LOGGER.debug("create stop script")
    with open("sge_stop", "w") as fp:
        fp.write(templates.get_template("sge_epilogue").render(job_id=job_id))
    os.chmod("sge_stop", 0o755)
    LOGGER.debug("done")

    # write epilogue script
    LOGGER.debug("create epilogue script")
    with open("sge_epilogue", "w") as fp:
        fp.write(templates.get_template("sge_epilogue").render(job_id=job_id))
    os.chmod("sge_epilogue", 0o755)
    LOGGER.debug("done")
