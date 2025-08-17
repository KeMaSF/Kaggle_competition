


import numpy as np


from scipy.optimize import minimize

# parameters
max_q = 9

# Obj function
def final_score(param):
    q1, q2, q3 = param
    avg_q = (q1 + q2 + q3) / 3
    var_q = np.var([q1, q2, q3])
    score = var_q / (max_q - avg_q)
    return -score # minimization problem

initial = [1.0, 2.0, 3.0]

bounds = [(0, max_q), (0, max_q), (0, max_q)]

result = minimize(final_score, initial, bounds = bounds)
o_q1, o_q2, o_q3 = result.x
max_score = -result.fun
print(o_q1, o_q2, o_q3, max_score)





