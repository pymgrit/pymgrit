from firedrake_heat_equation import vector_standard
from firedrake import  *
import pickle
import numpy as np
import dill
import json

def get_unpicklable(instance, exception=None, string='', first_only=True):
    """
    Recursively go through all attributes of instance and return a list of whatever
    can't be pickled.

    Set first_only to only print the first problematic element in a list, tuple or
    dict (otherwise there could be lots of duplication).
    """
    problems = []
    if isinstance(instance, tuple) or isinstance(instance, list):
        for k, v in enumerate(instance):
            try:
                pickle.dumps(v)
            except BaseException as e:
                problems.extend(get_unpicklable(v, e, string + f'[{k}]'))
                if first_only:
                    break
    elif isinstance(instance, dict):
        for k in instance:
            try:
                pickle.dumps(k)
            except BaseException as e:
                problems.extend(get_unpicklable(
                    k, e, string + f'[key type={type(k).__name__}]'
                ))
                if first_only:
                    break
        for v in instance.values():
            try:
                pickle.dumps(v)
            except BaseException as e:
                problems.extend(get_unpicklable(
                    v, e, string + f'[val type={type(v).__name__}]'
                ))
                if first_only:
                    break
    else:
        for k, v in instance.__dict__.items():
            try:
                pickle.dumps(v)
            except BaseException as e:
                problems.extend(get_unpicklable(v, e, string + '.' + k))

    # if we get here, it means pickling instance caused an exception (string is not
    # empty), yet no member was a problem (problems is empty), thus instance itself
    # is the problem.
    if string != '' and not problems:
        problems.append(
            string + f" (Type '{type(instance).__name__}' caused: {exception})"
        )

    return problems

n = 20
mesh = PeriodicSquareMesh(n, n, 10)
V = FunctionSpace(mesh, "DG", 1)
function_space = V
sol = vector_standard.Vector(V)

print(V.__dict__)
print('-----------------------------------------')
print(sol.vec.__dict__)

x = SpatialCoordinate(mesh)
initial_tracer = exp(-((x[0] - 5) ** 2 + (x[1] - 5) ** 2))
sol.vec.interpolate(initial_tracer)

print(sol.norm())

#print(sol.vec.__dict__)
function_dat = sol.vec.dat
a = np.copy(sol.vec.dat)
function_vector = sol.vec.vector()
for i in range(len(sol.vec.dat.data)):
    sol.vec.dat.data[i] = i
print(function_dat.data)
print(function_vector)

#test = Function(function_vector)

# sol.vec.dat.data = sol.vec.dat.data
# print(sol.norm())
pickle.dump(sol.vec, open( "save.p", "wb" ) )
#dill.dump( sol, open( "save.p", "wb" ) )
#print(V.__dict__)
#print(get_unpicklable(V))
#dill.detect.baditems(sol)
#dill.detect.trace(True)
#dill.detect.errors(sol)
#dill.dump( sol, open( "save.p", "wb" ) )