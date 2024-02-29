
import copy
import svgwrite

from xml.dom import minidom
from svg.path import parse_path

from .dicts_converter import DotsDictsConverter 
from .vector_converter import DotsVectorConverter 


class Converter:
    def __init__(self, vector_len=10):
        self.dots_dicts_converter = DotsDictsConverter()
        self.dots_vector_converter = DotsVectorConverter(vector_len)

    def to_vector(self, svg, is_one_len=True):
        dots = svg['d']

        dots_dicts = self.dots_dicts_converter.convert(dots, svg['width'], svg['height'])

        vector = self.dots_vector_converter.convert(dots_dicts, is_one_len)
        
        return vector

    def to_svg(self, vector):
        dots_dicts = self.dots_vector_converter.reconvert(vector)

        dots = self.dots_dicts_converter.reconvert(dots_dicts)

        width, height = self.dots_vector_converter.get_size(dots_dicts)

        svg = {
            'width': width,
            'height': height,
            'd': dots,
        }

        return svg

    def open(self, path):
        doc = minidom.parse(path)

        path = doc.getElementsByTagName('path')[0]
        d = path.getAttribute('d')

        svg = doc.getElementsByTagName('svg')

        try:
            width = float(svg[0].getAttribute('width').replace('px', ''))
            height = float(svg[0].getAttribute('height').replace('px', ''))
        
        except:
            width, height = [float(i) for i in (svg[0].getAttribute('viewBox')).split()][2:4]

        d = parse_path(d)

        svg = {
            'width': width,
            'height': height,
            'd': d,
        }

        return svg

    def save(self, path, svg):
        dwg = svgwrite.Drawing(
            path,
            profile='tiny',
            fill_rule="evenodd", 
            size=(f"{svg['width']}px", f"{svg['height']}px")
        )
        
        dwg.add(
            dwg.path( d=svg['d'].d(),
            fill="#000")
        )
        
        dwg.save()
