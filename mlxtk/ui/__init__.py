import pkg_resources
from PyQt5 import uic


def load_ui(name):
    path = pkg_resources.resource_filename(__name__, name)
    return uic.loadUi(path)


# PySide2 version
# def load_ui(name):
#     path = pkg_resources.resource_filename(__name__, name)
#     f = QFile(path)
#     f.open(QFile.ReadOnly)
#     loader = QUiLoader()
#     window = loader.load(f)
#     f.close()
#     return window


def replace_widget(widget, new_widget):
    widget.parentWidget().layout().replaceWidget(widget, new_widget)
    widget.hide()
    new_widget.show()
