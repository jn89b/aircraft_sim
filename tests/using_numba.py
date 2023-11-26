"""
https://www.youtube.com/watch?v=x58W9A2lnQc&ab_channel=JackofSome
"""
from numba import jit
import random 
#time the function
import time

def monte_carlo_pi(nsamples) -> float:
    acc = 0
    for i in range(nsamples):
        x = random.random()
        y = random.random()
        if x**2 + y**2 < 1.0:
            acc += 1
    return 4.0*acc/nsamples

num_samples = 10000


time_start = time.time()
pi = monte_carlo_pi(num_samples)
print("time elapsed:", time.time() - time_start)
print("pi:", pi)


#use jit this is very slow at first intialization
monte_carlo_pi_jit = jit()(monte_carlo_pi)
time_start = time.time()
pi_jit = monte_carlo_pi_jit(num_samples)
print("time elapsed for jit:", time.time() - time_start)
print("pi:", pi_jit)

#after first initialization it is much faster
time_start = time.time()
pi_jit = monte_carlo_pi_jit(num_samples)
print("time elapsed after initi jit:", time.time() - time_start)
print("pi:", pi_jit)