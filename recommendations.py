# A dictionary of movie critics and their ratings of a small
# set of movies
from math import sqrt

critics = {
    'Lisa Rose': {
        'Lady in the Water': 2.5,
        'Snakes on a Plane': 3.5,
        'Just My Luck': 3.0,
        'Superman Returns': 3.5,
        'You, Me and Dupree': 2.5,
        'The Night Listener': 3.0
    }, 'Gene Seymour': {
        'Lady in the Water': 3.0,
        'Snakes on a Plane': 3.5,
        'Just My Luck': 1.5,
        'Superman Returns': 5.0,
        'The Night Listener': 3.0,
        'You, Me and Dupree': 3.5
    },    'Michael Phillips': {
        'Lady in the Water': 2.5,
        'Snakes on a Plane': 3.0,
        'Superman Returns': 3.5,
        'The Night Listener': 4.0
    }, 'Claudia Puig': {
        'Snakes on a Plane': 3.5,
        'Just My Luck': 3.0,
        'The Night Listener': 4.5,
        'Superman Returns': 4.0,
        'You, Me and Dupree': 2.5
    },    'Mick LaSalle': {
        'Lady in the Water': 3.0,
        'Snakes on a Plane': 4.0,
        'Just My Luck': 2.0,
        'Superman Returns': 3.0,
        'The Night Listener': 3.0,
        'You, Me and Dupree': 2.0
    }, 'Jack Matthews': {
        'Lady in the Water': 3.0,
        'Snakes on a Plane': 4.0,
        'The Night Listener': 3.0,
        'Superman Returns': 5.0,
        'You, Me and Dupree': 3.5
    }, 'Toby': {
        'Snakes on a Plane': 4.5,
        'You, Me and Dupree': 1.0,
        'Superman Returns': 4.0
    }
}

# Euclidean Distance Score


def sim_distance(prefs, a, b):
    apref, bpref = prefs[a], prefs[b]  # save reference
    si = set.intersection(set(apref), set(bpref))
    if len(si) == 0:
        return 0
    sum_of_squares = sum(map(lambda x: (apref[x] - bpref[x]) ** 2, si))
    return 1 / (1 + sum_of_squares)

sim_distance(critics, 'Lisa Rose', 'Gene Seymour')

# Pearson Correlation Score


def sim_pearson(prefs, a, b):
    apref, bpref = prefs[a], prefs[b]  # save reference
    si = set.intersection(set(apref), set(bpref))
    n = len(si)
    if n == 0:
        return 0
    suma, sumb = map(lambda x: sum(x[i] for i in si), (apref, bpref))
    ssqa, ssqb = map(lambda x: sum(x[i] ** 2 for i in si), (apref, bpref))
    num = sum(apref[i] * bpref[i] for i in si) - (suma * sumb / n)
    den = sqrt((ssqa - suma ** 2 / n) * (ssqb - sumb ** 2 / n))
    return num / den

sim_pearson(critics, 'Lisa Rose', 'Gene Seymour')


def most_similar(prefs, person, n=5, measure=sim_pearson):
    scores = sorted(((measure(prefs, person, x), x)
                     for x in prefs if x != person), reverse=True)
    return scores[:n]


most_similar(critics, 'Toby', n=3)


def get_recommendations(prefs, person, measure=sim_pearson):
    this_pref = prefs[person]
    total_score, total_sim = {}, {}

    for other in prefs:
        if other == person:
            continue  # ignore himself
        sim = measure(prefs, person, other)
        if sim <= 0:
            continue  # nothing in common
        for item in prefs[other]:
            if item in this_pref and this_pref[item] != 0:  # watched
                continue

            total_score.setdefault(item, 0.0)
            total_sim.setdefault(item, 0.0)
            total_score[item] += sim * prefs[other][item]
            total_sim[item] += sim

    return sorted(((total / total_sim[item], item)
                   for item, total in total_score.items()),
                  reverse=True)


get_recommendations(critics, 'Toby')
get_recommendations(critics, 'Toby', sim_distance)


def by_item(prefs):
    result = {}
    for name in prefs:
        for item in prefs[name]:
            result.setdefault(item, {})
            result[item][name] = prefs[name][item]
    return result

movies = by_item(critics)
most_similar(movies, 'Superman Returns')
get_recommendations(movies, 'Just My Luck')


def find_similar_items(prefs, n=10):
    """Precompute the top n similar items for each item,
    needs to be update periodically.

    Parameters
    ----------
    prefs: {'user1': {'item1': score,
                      'item2': score, ...},
            'user2': {...}
    ...}
    n: top n

    Return
    -----------
    {'item1': [(similariy, 'similar-item1'),
              (similarity, 'similar-item2'), ...]
     'item2': [...]
    ...}
    """
    item_prefs = by_item(prefs)
    return {item: most_similar(item_prefs, item,
                               n=n, measure=sim_distance)
            for item in item_prefs}
