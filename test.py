# from itertools import cycle
import numpy as np

h, w = 4, 4
input = np.arange(1,17).reshape([h,w])

output = []
cycle_num = h//2
for i_cycle in range(cycle_num):
    for line in range(4):
        if line == 0:
            i = i_cycle
            for j in range(i_cycle, w - i_cycle -1):
                output.append(input[i, j]) 
            # import pdb; pdb.set_trace()
        if line == 1:
            j = w - i_cycle - 1
            for i in range(i_cycle, h - i_cycle -1):
                output.append(input[i, j])
            # import pdb; pdb.set_trace()
        if line == 2:
            i = h - i_cycle - 1
            for j in range(w - i_cycle -1, i_cycle, -1):
                output.append(input[i, j])
            # import pdb; pdb.set_trace()
        if line == 3:
            j = i_cycle
            for i in range(h - i_cycle -1, i_cycle, -1):
                output.append(input[i, j])
            # import pdb; pdb.set_trace()
        
print(output)