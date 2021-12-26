import pandas as pd
import numpy as np
import os
import scipy.stats as st


def friedman_test(*args):
    k = len(args)
    if k < 2: raise ValueError('Less than 2 levels')
    n = len(args[0])
    if len(set([len(v) for v in args])) != 1: raise ValueError('Unequal number of samples')

    rankings = []
    for i in range(n):
        row = [col[i] for col in args]
        row_sort = sorted(row)
        rankings.append([row_sort.index(v) + 1 + (row_sort.count(v) - 1) / 2. for v in row])

    rankings_avg = [np.mean([case[j] for case in rankings]) for j in range(k)]
    rankings_cmp = [r / np.sqrt(k * (k + 1) / (6. * n)) for r in rankings_avg]

    chi2 = ((12 * n) / float((k * (k + 1)))) * (
                (sum(r ** 2 for r in rankings_avg)) - ((k * (k + 1) ** 2) / float(4)))
    iman_davenport = ((n - 1) * chi2) / float((n * (k - 1) - chi2))

    p_value = 1 - st.f.cdf(iman_davenport, k - 1, (k - 1) * (n - 1))

    return iman_davenport, p_value, rankings_avg, rankings_cmp


def holm_test(ranks, control=None):
    k = len(ranks)
    values = list(ranks.values())
    keys = list(ranks.keys())
    if not control:
        control_i = values.index(min(values))
    else:
        control_i = keys.index(control)

    comparisons = [keys[control_i] + " vs " + keys[i] for i in range(k) if i != control_i]
    z_values = [abs(values[control_i] - values[i]) for i in range(k) if i != control_i]
    p_values = [2 * (1 - st.norm.cdf(abs(z))) for z in z_values]
    # Sort values by p_value so that p_0 < p_1
    p_values, z_values, comparisons = map(list, zip(*sorted(zip(p_values, z_values, comparisons), key=lambda t: t[0])))
    adj_p_values = [min(max((k - (j + 1)) * p_values[j] for j in range(i + 1)), 1) for i in range(k - 1)]

    return comparisons, z_values, p_values, adj_p_values
