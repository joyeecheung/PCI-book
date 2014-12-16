from collections import Counter, defaultdict
from math import log

RESULT_IDX = -1
SEP = '\t'
MISSING = None


def clean(token):
    token = token.strip()
    try:
        return int(token)
    except ValueError:
        return None if token == 'None' else token


my_data = [map(clean, line.split(SEP))
           for line in file('decision_tree_example.txt')]


def divide(data, tests):
    return [filter(lambda x: test(x), data) for test in tests]


def divide_bool(data, attr, value):
    counter = Counter(record[attr] for record in data)
    guess = counter.most_common()[0][0]

    fix_missing = lambda x: guess if x == MISSING else x

    if isinstance(value, int) or isinstance(value, float):
        left = lambda x: fix_missing(x[attr]) >= value
        right = lambda x: fix_missing(x[attr]) < value
    else:
        left = lambda x: fix_missing(x[attr]) == value
        right = lambda x: fix_missing(x[attr]) != value

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
                 left=None, right=None, leaves=None, count=None):
        """"""
        self.attr = attr
        self.value = value
        self.leaves = leaves  # the classification or regression
        self.left = left
        self.right = right
        self.count = count


def build_tree(data, score=entropy):
    # empty data
    if len(data) == 0:
        return Node()

    current_score = score(data)
    best_gain, best_criteria, best_sets = 0.0, None, None

    attr_list = list(xrange(len(data[0])))
    del attr_list[RESULT_IDX]
    for attr in attr_list:
        values = set(record[attr] for record in data).remove(MISSING)
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
        return Node(best_criteria[0], best_criteria[1],
                    left, right, count=len(data))
    else:  # all tested
        return Node(leaves=count(data), count=len(data))


def print_tree(tree, indent=''):
    # Is this a leaf node?
    if tree.leaves:
        print str(tree.leaves)
    else:
        # Print the criteria
        print str(tree.attr) + ':' + str(tree.value) + '? ', tree.count
        # Print the branches
        print indent + 'T->',
        print_tree(tree.left, indent + '   ')
        print indent + 'F->',
        print_tree(tree.right, indent + '   ')


# def classify(tree, observation):
#     if tree.leaves:
#         return tree.leaves

#     value = observation[tree.attr]
#     if isinstance(value, int) or isinstance(value, float):
#         branch = tree.left if value >= tree.value else tree.right
#     else:
#         branch = tree.left if value == tree.value else tree.right
#     return classify(branch, observation)


def classify(tree, observation):
    if tree.leaves:
        return tree.leaves

    value = observation[tree.attr]

    if value == MISSING:
        probe_left = classify(tree.left, observation)
        probe_right = classify(tree.right, observation)

        left_weight = float(tree.left.count) / tree.count
        right_weight = float(tree.right.count) / tree.count
        result = defaultdict(int)
        for k, v in probe_left.items():
            result[k] += v * left_weight
        for k, v in probe_right.items():
            result[k] += v * right_weight

        return dict(result)

    if isinstance(value, int) or isinstance(value, float):
        branch = tree.left if value >= tree.value else tree.right
    else:
        branch = tree.left if value == tree.value else tree.right
    return classify(branch, observation)


def mdclassify(observation, tree):
    if tree.leaves != MISSING:
        return tree.leaves
    else:
        v = observation[tree.attr]
        if v == MISSING:
            tr, fr = mdclassify(observation, tree.left), mdclassify(
                observation, tree.right)
            tcount = sum(tr.values())
            fcount = sum(fr.values())
            tw = float(tcount) / (tcount + fcount)
            fw = float(fcount) / (tcount + fcount)
            result = {}
            for k, v in tr.items():
                result[k] = v * tw
            for k, v in fr.items():
                result[k] = v * fw
            return result
        else:
            if isinstance(v, int) or isinstance(v, float):
                if v >= tree.value:
                    branch = tree.left
                else:
                    branch = tree.right
            else:
                if v == tree.value:
                    branch = tree.left
                else:
                    branch = tree.right
            return mdclassify(observation, branch)
