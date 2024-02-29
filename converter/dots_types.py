
from svg.path.path import Line, CubicBezier, QuadraticBezier, Move


class DotTypes:
    def __init__(self):
        self.CubicBezier = CubicBezier
        self.QuadraticBezier = QuadraticBezier
        self.Line = Line
        self.Move = Move

        self.names = [
            'CubicBezier',
            'QuadraticBezier',
            'Line',
            'Move',
        ]

    def get(self, dot_name):
        return getattr(self, dot_name)

    def has(self, dot_name):
        return hasattr(self, dot_name)

    def get_type(self, dot_name):
        for i, name in enumerate(self.names):
            if dot_name == name:
                return i

    def get_rep_by_type(self, dot_type):
        return self.get(self.names[dot_type])

    def get_names(self):
        return self.names