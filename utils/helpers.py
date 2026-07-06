import math


def euclidean_distance(p1, p2):
    return math.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2)


def clamp(value, min_val, max_val):
    return max(min_val, min(value, max_val))
