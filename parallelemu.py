import multiprocessing
from multiprocessing import Pool
import linps


def ps(n):
    ps = linps
    return ps

if __name__ == '__main__':
    #Change n to vary number input
    n = range(2)
    p = Pool()
    result = p.map(ps, n)
    print(result)
    p.close()
    p.join()


cpu_count = multiprocessing.cpu_count()
print(cpu_count)
