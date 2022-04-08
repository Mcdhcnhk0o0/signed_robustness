from .frustration import Frustration
from .local_search import LocalSearch
from .neighborhood import Neighborhood
from .initialization import Initialization
from .objective_function import ObjectiveFunction


"""
The data structure of this program is agreed here.
1. The solution of the problem is described by a list or a dict, sometimes a numpy.ndarray.
    * A list is the best.
    example: self.solution = {i: i for i in range(n)}
             self.solution = [0] * n
2. The partition is described by a dict with numbers as keys and sets as values.
    example: self.partition = {0: {0, 1, 2}, 1: {3, 4, 5}, 2: {6}}
3. The dataset is represented by a two-dimensional default dict.
    example: defaultdict(lambda: defaultdict(lambda: 0))
"""
