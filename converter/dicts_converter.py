
from .dots_types import DotTypes
from svg.path.path import  Path
from svg.path.path import Line, CubicBezier, QuadraticBezier
from svg.path import Close, Move

class DotsDictsConverter:
    def __init__(self):
        self.dots_types = DotTypes()

        self.control_names = [
            'control',
            'control1',
            'control2',
        ]

        self.width = None
        self.height = None

    def get_width(self, dots):
        min_coord = float('inf')
        max_coord = float('-inf')

        for dot in dots:
            x, y = self.get_int_coord(dot.start)
            min_coord = min(min_coord, x)
            max_coord = max(max_coord, x)

            x, y = self.get_int_coord(dot.end)
            min_coord = min(min_coord, x)
            max_coord = max(max_coord, x)

        self.x_bias = min_coord

        return max_coord - min_coord

    def get_height(self, dots):
        min_coord = float('inf')
        max_coord = float('-inf')

        for dot in dots:
            x, y = self.get_int_coord(dot.start)
            min_coord = min(min_coord, y)
            max_coord = max(max_coord, y)

            x, y = self.get_int_coord(dot.end)
            min_coord = min(min_coord, y)
            max_coord = max(max_coord, y)

        self.y_bias = min_coord

        return max_coord - min_coord

    def convert(self, dots, width, height):
        dots_dicts = []

        self.width = self.get_width(dots)
        self.height = self.get_height(dots)

        for i, dot in enumerate(dots):
            if self.has_dot_rep(dot):
                dot_dict = self.dot_to_dict(dot)

                dots_dicts.append(dot_dict)

        return dots_dicts

    def get_dot_controls(self, dot):
        controls = []
        for control_name in self.control_names:
            if hasattr(dot, control_name):
                control_coord = getattr(dot, control_name)
                controls.append(self.convert_coord(control_coord))

        return controls

    def dot_to_dict(self, dot):
        dot_dict = {}

        dot_dict['s'] = self.convert_coord(dot.start)   
        dot_dict['e'] = self.convert_coord(dot.end)

        dot_dict['cs'] = self.get_dot_controls(dot)

        dot_dict['type'] = self.get_dot_type(dot)

        return dot_dict

    def get_dot_type(self, dot):
        dot_rep_name = self.get_rep_name(dot)
        dot_type = self.dots_types.get_type(dot_rep_name)
        return dot_type

    def has_dot_rep(self, dot):
        dot_rep_name = self.get_rep_name(dot)
        has = self.dots_types.has(dot_rep_name)
        return has

    def get_rep_name(self, dot):
        dot_rep_name = type(dot).__name__
        return dot_rep_name

    def convert_coord(self, coord):
        x, y = self.get_int_coord(coord)

        maxwh = max(self.width, self.height)

        x -= self.x_bias
        y -= self.y_bias
        
        x /= maxwh
        y /= maxwh

        return [x, y]

    def get_int_coord(self, cord):
        return (cord.real, cord.imag)

    def reconvert(self, dots_dicts):
        dots_reps = self.to_dots_reps(dots_dicts)

        dots = Path(*dots_reps)
        dots.insert(0, Move(to=dots[0].start))
        dots.append(Close(start=dots[-1].end, end=dots[-1].end))

        return dots

    def plus_one(self, coord):
        return [coord[0], coord[1]]
    
    def to_dots_reps(self, dots_dicts):
        dots_reps = []
        for dot_dict in dots_dicts:
            dot_dict['s'] = self.plus_one(dot_dict['s'])
            dot_dict['e'] = self.plus_one(dot_dict['e'])
            dot_dict['cs'] = [self.plus_one(i) for i in dot_dict['cs']]

            rep = self.dots_types.get_rep_by_type(dot_dict['type'])
            options = []

            options.append(complex(*dot_dict['s']))

            for i in range(len(dot_dict['cs'])):
                dot_dict['cs'][i] = complex(*dot_dict['cs'][i])

            options.extend(dot_dict['cs'])

            options.append(complex(*dot_dict['e']))

            try:
                rep_obj = rep(*options)
            except:
                rep_obj = rep(*options[:-1])
            
            dots_reps.append(rep_obj)

        return dots_reps
