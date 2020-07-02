from matplotlib.axes import Axes
from matplotlib.lines import Line2D


class Cursor1D:
    def __init__(self, axes: Axes, is_xcursor: bool):
        self.axes = axes
        self.is_xcursor = is_xcursor
        self.line = None  # type: Line2D
        self.line_style = ":"  # type: str
        self.color = "red"

    def set_position(self, p: float):
        if self.line is None:
            if self.is_xcursor:
                self.line = self.axes.axvline(p, ls=self.line_style, color=self.color)
            else:
                self.line = self.axes.axhline(p, ls=self.line_style, color=self.color)
        else:
            if self.is_xcursor:
                self.line.set_xdata([p for _ in self.line.get_xdata()])
            else:
                self.line.set_ydata([p for _ in self.line.get_ydata()])

    def set_line_style(self, style: str):
        self.line_style = style
        if self.line:
            self.line.set_style(style)

    def set_color(self, color):
        self.color = color
        if self.line:
            self.line.set_color(color)


class CursorX(Cursor1D):
    def __init__(self, axes: Axes):
        super().__init__(axes, True)

    def set_x(self, x: float):
        self.set_position(x)


class CursorY(Cursor1D):
    def __init__(self, axes: Axes):
        super().__init__(axes, False)

    def set_y(self, y: float):
        self.set_position(y)


class CursorXY:
    def __init__(self, axes: Axes):
        self.axes = axes
        self.cursor_x = CursorX(axes)
        self.cursor_y = CursorY(axes)

    def set_x(self, x: float):
        self.cursor_x.set_position(x)

    def set_y(self, y: float):
        self.cursor_y.set_position(y)

    def set_line_style(self, style: str):
        self.cursor_x.set_line_style(style)
        self.cursor_y.set_line_style(style)

    def set_color(self, color):
        self.cursor_x.set_color(color)
        self.cursor_y.set_color(color)
