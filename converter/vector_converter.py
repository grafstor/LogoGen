
import math
import copy

import numpy as np
from .dots_types import DotTypes


class DotsVectorConverter:
    def __init__(self, vector_len=10, scale=100):

        self.vector_len = vector_len

        self.scale = scale

        self.dots_types = DotTypes()

    def convert(self, dots_dicts, is_one_len=True):
        
        vector = []

        dots_dicts = self.fill_moves(dots_dicts)
        past_angle = None

        for i, dot_dict in enumerate(dots_dicts):
            # dot_vector = self.dot_dict_to_dot_vector(dot_dict)
            dot_vector, past_angle = self.dot_dict_to_dot_vector(dot_dict, past_angle)

            vector.append(dot_vector)

        if is_one_len:
            self.fill_vector_nulls(vector)
            vector = vector[:self.vector_len]     

        return vector

    def reconvert(self, vector, scale):
        self.scale = scale
        dots_dicts = []

        vector = self.remove_nulls_vectors(vector)

        next_root_coord = [0, 0]

        min_x_coord = float("inf")
        min_y_coord = float("inf")

        angle = 0 

        for dot_vector in vector:
            # dot_dict, next_root_coord = self.dot_vector_to_dot_dict(dot_vector, next_root_coord)
            dot_dict, next_root_coord, angle = self.dot_vector_to_dot_dict(dot_vector, next_root_coord, angle)

            min_x_coord = min(next_root_coord[0], min_x_coord)
            min_y_coord = min(next_root_coord[1], min_y_coord)

            dots_dicts.append(dot_dict)

        dots_dicts = self.normalize_coords(dots_dicts, min_x_coord, min_y_coord)
        dots_dicts = self.scale_coords(dots_dicts, self.scale)

        #dots_dicts = self.remove_moves(dots_dicts)

        return dots_dicts

    def get_size(self, dots_dicts):
        max_x_coord = float("-inf")
        max_y_coord = float("-inf")

        for dot_dict in dots_dicts:
            max_x_coord = max(dot_dict['s'][0], max_x_coord)
            max_y_coord = max(dot_dict['s'][1], max_y_coord)

            max_x_coord = max(dot_dict['e'][0], max_x_coord)
            max_y_coord = max(dot_dict['e'][1], max_y_coord)

        return max_x_coord, max_y_coord

    def scale_coords(self, dots_dicts, scale):
        for dot_dict in dots_dicts:
            dot_dict['s'] = [dot_dict['s'][0]*scale, dot_dict['s'][1]*scale]
            dot_dict['e'] = [dot_dict['e'][0]*scale, dot_dict['e'][1]*scale]

            if dot_dict['type'] == 0:
                dot_dict['cs'][0] = [dot_dict['cs'][0][0]*scale, dot_dict['cs'][0][1]*scale]
                dot_dict['cs'][1] = [dot_dict['cs'][1][0]*scale, dot_dict['cs'][1][1]*scale]

            elif dot_dict['type'] == 1:
                dot_dict['cs'] = [dot_dict['cs'][0]*scale, dot_dict['cs'][1]*scale]

        return dots_dicts

    def normalize_coords(self, dots_dicts, min_x_coord, min_y_coord):        
        for dot_dict in dots_dicts:
            dot_dict['s'] = [dot_dict['s'][0]-min_x_coord, dot_dict['s'][1]-min_y_coord]
            dot_dict['e'] = [dot_dict['e'][0]-min_x_coord, dot_dict['e'][1]-min_y_coord]

            if dot_dict['type'] == 0:
                dot_dict['cs'][0] = [dot_dict['cs'][0][0]-min_x_coord, dot_dict['cs'][0][1]-min_y_coord]
                dot_dict['cs'][1] = [dot_dict['cs'][1][0]-min_x_coord, dot_dict['cs'][1][1]-min_y_coord]

            elif dot_dict['type'] == 1:
                dot_dict['cs'] = [dot_dict['cs'][0]-min_x_coord, dot_dict['cs'][1]-min_y_coord]

        return dots_dicts

    def fill_moves(self, dots_dicts):

        new_dots_dicts = []

        for i in range(len(dots_dicts)-1):
            new_dots_dicts.append(dots_dicts[i])

            if dots_dicts[i]['e'] != dots_dicts[i+1]['s']:
                move_dot = {}
                move_dot['s'] = dots_dicts[i]['e']
                move_dot['e'] = dots_dicts[i+1]['s']
                move_dot['cs'] = []
                move_dot['type'] = 3

                new_dots_dicts.append(move_dot)

        new_dots_dicts.append(dots_dicts[-1])

        return new_dots_dicts

    def remove_moves(self, dots_dicts):

        new_dots_dicts = []

        for i in range(len(dots_dicts)):
            if dots_dicts[i]['type'] != 3:
                new_dots_dicts.append(dots_dicts[i])

        return new_dots_dicts

    # def dot_dict_to_dot_vector(self, dot_dict):
    #     dot_vector = []

    #     next_root_coord = dot_dict['s']

    #     dot_vector += self.subtract_coords(next_root_coord, dot_dict['e']) # 1, 2

    #     for control in dot_dict['cs']:
    #         dot_vector += self.subtract_coords(next_root_coord, control) # 3, 4  / 3, 4, 5, 6

    #     dot_vector += self.get_nulls(dot_vector, min_len=6) # 5,6 

    #     dot_vector += self.type_to_onehot(dot_dict['type']) # 

    #     dot_vector += [-1]

    #     return dot_vector

    # def dot_vector_to_dot_dict(self, dot_vector, next_root_coord):
    #     dot_dict = {}

    #     dot_dict['s'] = next_root_coord
    #     new_next_root_coord = self.add_up_coords(next_root_coord, dot_vector[:2])
    #     dot_dict['e'] = new_next_root_coord 

    #     dot_dict['cs'] = []
        
    #     dot_dict['type'] = np.argmax(np.array(dot_vector[6:10]))

    #     if dot_dict['type'] == 0:
    #         dot_dict['cs'].append(self.add_up_coords(next_root_coord, dot_vector[2:4]))            
    #         dot_dict['cs'].append(self.add_up_coords(next_root_coord, dot_vector[4:6]))

    #     elif dot_dict['type'] == 1:
    #         dot_dict['cs'].append(self.add_up_coords(next_root_coord, dot_vector[2:4]))


    #     return dot_dict, new_next_root_coord


    # def dot_dict_to_dot_vector(self, dot_dict):
    #     dot_vector = []

    #     dot_vector += dot_dict['e']

    #     # dot_vector += self.subtract_coords(next_root_coord, dot_dict['e']) # 1, 2

    #     for control in dot_dict['cs']:
    #         dot_vector += control # 3, 4  / 3, 4, 5, 6

    #     dot_vector += self.get_nulls(dot_vector, min_len=6) # 5,6 / 7, 8

    #     dot_vector += self.type_to_onehot(dot_dict['type']) # 

    #     dot_vector += [-1]

    #     return dot_vector

    # def dot_vector_to_dot_dict(self, dot_vector, next_root_coord):
    #     dot_dict = {}

    #     dot_dict['s'] = next_root_coord
    #     new_next_root_coord = dot_vector[:2]
    #     dot_dict['e'] = new_next_root_coord

    #     dot_dict['cs'] = []
        
    #     dot_dict['type'] = np.argmax(np.array(dot_vector[6:10]))

    #     if dot_dict['type'] == 0:
    #         dot_dict['cs'].append(dot_vector[2:4])            
    #         dot_dict['cs'].append(dot_vector[4:6])

    #     elif dot_dict['type'] == 1:
    #         dot_dict['cs'].append(dot_vector[2:4])


    #     return dot_dict, new_next_root_coord

    def get_distance(self, c1, c2):
        return ((c1[0] - c2[0])**2 + (c1[1] - c2[1])**2)**0.5

    def get_angle(self, coord1, coord2):
        # Вычисляем угол в радианах
        delta_y = coord2[1] - coord1[1]
        delta_x = coord2[0] - coord1[0]
        angle_rad = math.atan2(delta_y, delta_x)
        
        # Нормализуем угол в диапазоне [0, 2π]
        if angle_rad < 0:
            angle_rad += 2 * math.pi
        
        # Нормализуем угол от 0 до 1
        normalized_angle = angle_rad / (2 * math.pi)
        
        return normalized_angle

    def normalized_angle_length_to_vector(self, normalized_angle, length):
        """
        Преобразует нормализованный угол и длину в координаты вектора (x, y).

        Args:
        normalized_angle: Нормализованный угол в радианах (от 0 до 1).
        length: Длина вектора.

        Returns:
        Координаты вектора (x, y).
        """
        angle_radians = normalized_angle * 2 * math.pi
        x = length * math.cos(angle_radians)
        y = length * math.sin(angle_radians)
        return [x, y]

    def dot_dict_to_dot_vector(self, dot_dict, past_angle):
        dot_vector = []

        next_root_coord = dot_dict['s']

        dot_vector += [self.get_distance(next_root_coord, dot_dict['e'])]

        current_angle = self.get_angle(next_root_coord, dot_dict['e'])
        # print(current_angle)
        if past_angle == None:
            d_angle = current_angle
        else:
            d_angle = current_angle - past_angle 
            
        past_angle = copy.copy(current_angle)

        dot_vector += [d_angle]
        # dot_vector += self.subtract_coords(next_root_coord, dot_dict['e']) # 1, 2

        for control in dot_dict['cs']:
            dot_vector += self.subtract_coords(next_root_coord, control) # 3, 4  / 3, 4, 5, 6

        dot_vector += self.get_nulls(dot_vector, min_len=6) # 5,6 

        dot_vector += self.type_to_onehot(dot_dict['type']) # 

        dot_vector += [0]

        return dot_vector, past_angle

    def dot_vector_to_dot_dict(self, dot_vector, next_root_coord, angle):
        dot_dict = {}

        dot_dict['s'] = next_root_coord
        vec_len = dot_vector[0]
        d_angle = dot_vector[1]
        angle += d_angle

        new_next_root_coord = self.add_up_coords(next_root_coord, self.normalized_angle_length_to_vector(angle, vec_len))
        dot_dict['e'] = new_next_root_coord 

        dot_dict['cs'] = []
        
        dot_dict['type'] = np.argmax(np.array(dot_vector[6:10]))

        if dot_dict['type'] == 0:
            dot_dict['cs'].append(self.add_up_coords(next_root_coord, dot_vector[2:4]))            
            dot_dict['cs'].append(self.add_up_coords(next_root_coord, dot_vector[4:6]))

        elif dot_dict['type'] == 1:
            dot_dict['cs'].append(self.add_up_coords(next_root_coord, dot_vector[2:4]))


        return dot_dict, new_next_root_coord, angle
        
    def type_to_onehot(self, dot_type):
        onehot = [0]*4
        onehot[dot_type] = 1
        return onehot

    def fill_vector_nulls(self, vector):
        vector_len = len(vector)
        vector_len_part = len(vector[0])
        null_vector_part = [0]*(vector_len_part - 1) + [1]

        for _ in range(self.vector_len - vector_len):
            vector.append(null_vector_part)

    def remove_nulls_vectors(self, vector):
        new_vector = []
        for i in vector:
            if i[-1] > 0.5:
                return new_vector
            new_vector.append(i)
        return new_vector

    def subtract_coords(self, co2, co1):
        return [co1[0] - co2[0], co1[1] - co2[1]]
    
    def add_up_coords(self, co2, co1):
        return [co1[0] + co2[0], co1[1] + co2[1]]

    def get_nulls(self, dot_vector, min_len):
        return [0 for _ in range(min_len - len(dot_vector))]
