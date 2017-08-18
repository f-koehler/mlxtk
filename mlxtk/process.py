import subprocess
import threading


def watch_process(cmd, stdout_watcher, stderr_watcher, **kwargs):
    kwargs["stdout"] = subprocess.PIPE
    kwargs["stderr"] = subprocess.PIPE
    kwargs["universal_newlines"] = True

    proc = subprocess.Popen(cmd, **kwargs)

    def watch_stdout(pipe):
        with pipe:
            for line in iter(pipe.readline, ""):
                stdout_watcher(line.strip("\n"))

    def watch_stderr(pipe):
        with pipe:
            for line in iter(pipe.readline, ""):
                stderr_watcher(line.strip("\n"))

    threading.Thread(target=watch_stdout, args=[proc.stdout]).start()
    threading.Thread(target=watch_stderr, args=[proc.stderr]).start()

    return_code = proc.wait()

    if return_code:
        raise subprocess.CalledProcessError("process failed", return_code)

    return return_code
