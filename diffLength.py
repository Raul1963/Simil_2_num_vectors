import time


def checkSameElems(l1: list, l2: list):
    f = 0
    # l1.sort()
    # l2.sort()
    # i = 0
    # j = 0
    # while i < len(l1) and j < len(l2):
    #     if l1[i] == l2[j]:
    #         f += 1
    #         i += 1
    #         j += 1
    #     elif l1[i] < l2[j]:
    #         i += 1
    #     else:
    #         j += 1
    mn=min(len(l1),len(l2))
    mx=max(len(l1),len(l2))
    for i in range(mn):
        if l1[i]==l2[i]:
            f+=1
    return f/mx


def checkSum(l1: list, l2: list):
    return min(sum(l1), sum(l2))/max(sum(l1), sum(l2))


def checkOrder(l1: list, l2: list):
    start1 = time.perf_counter()
    l1.sort()
    elapsed1 = time.perf_counter()-start1
    start2 = time.perf_counter()
    l2.sort()
    elapsed2 = time.perf_counter()-start2
    return elapsed1/elapsed2 if elapsed1 < elapsed2 else elapsed2/elapsed1


def compVect(l1: list, l2: list):
    if l1 == l2:
        return 1
    c1 = checkSameElems(l1, l2)
    c2 = checkSum(l1, l2)
    c3 = checkOrder(l1, l2)
    mn = min(len(l1), len(l2))
    mx = max(len(l1), len(l2))
    return mn/mx * ((c1+c2+c3)/3)


l1 = [1, 3, 4]
l2 = [3, 4, 6, 7]
print(compVect(l1, l2))
