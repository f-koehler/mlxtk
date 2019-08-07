import mlxtk


def task_plot_gpop():
    return mlxtk.doit_analyses.scan_plot_gpop("harmonic_trap_scan")


def task_plot_natpop():
    return mlxtk.doit_analyses.scan_plot_natpop("harmonic_trap_scan")
