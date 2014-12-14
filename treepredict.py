from collections import Counter
from math import log

RESULT_IDX = -1
SEP = '\t'


def clean(token):
    token = token.strip()
    try:
        return int(token)
    except ValueError:
        return token


my_data = [map(clean, line.split(SEP))
           for line in file('decision_tree_example.txt')]


def divide(data, tests):
    return [filter(lambda x: test(x), data) for test in tests]


def divide_bool(data, attr, value):
    if isinstance(value, int) or isinstance(value, float):
        left = lambda x: x[attr] >= value
        right = lambda x: x[attr] < value
    else:
        left = lambda x: x[attr] == value
        right = lambda x: x[attr] != value

    return (filter(left, data), filter(right, data))


def count(data):
    return Counter(record[RESULT_IDX] for record in data)


def gini_impurity(data):
    total, impurity = len(data), 0.0
    counter = count(data)

    for k1 in counter:
        p1 = float(counter[k1]) / total
        for k2 in counter:
            p2 = float(counter[k2]) / total if k1 != k2 else 0
            impurity += p1 * p2
    return impurity


def close(a, b, epsilon=6):
    return abs(a - b) <= 10 ** -epsilon


assert close(gini_impurity(my_data), 0.6328125)


def entropy(data):
    v = count(data).values()
    ent, total = 0.0, len(data)

    for vk in v:
        p = float(vk) / total
        ent -= p * log(p, 2)
    return ent


assert close(entropy(my_data), 1.5052408149441479)
set1, set2 = divide_bool(my_data, 2, 'yes')
assert close(entropy(set1), 1.2987949406953985)
assert close(gini_impurity(set1), 0.53125)


class Node(object):

    def __init__(self, attr=RESULT_IDX, value=None,
                 left=None, right=None, result=None):
        """"""
        self.attr = attr
        self.value = value
        self.result = result  # the classification or regression
        self.left = left
        self.right = right


def build_tree(data, score=entropy):
    # empty data
    if len(data) == 0:
        return Node()

    current_score = score(data)
    best_gain, best_criteria, best_sets = 0.0, None, None

    attr_list = list(xrange(len(data[0])))
    del attr_list[RESULT_IDX]
    for attr in attr_list:
        values = set(record[attr] for record in data)
        # find the attribute value resulted in best gain
        for value in values:
            set1, set2 = divide_bool(data, attr, value)
            p = float(len(set1)) / len(data)
            remainder = p * score(set1) + (1 - p) * score(set2)
            gain = current_score - remainder
            if (gain > best_gain and len(set1) > 0 and len(set2) > 0):
                best_gain, best_criteria = gain, (attr, value)
                best_sets = (set1, set2)

    # create subbranches
    if best_gain > 0:  # when there are more attribute to test
        left = build_tree(best_sets[0])
        right = build_tree(best_sets[1])
        return Node(best_criteria[0], best_criteria[1], left, right)
    else:  # all tested
        return Node(result=count(data))


def print_tree(tree, indent=''):
    # Is this a leaf node?
    if tree.result:
        print str(tree.result)
    else:
        # Print the criteria
        print str(tree.attr) + ':' + str(tree.value) + '? '
        # Print the branches
        print indent + 'T->',
        print_tree(tree.left, indent + '   ')
        print indent + 'F->',
        print_tree(tree.right, indent + '   ')
