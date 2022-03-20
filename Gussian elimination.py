import copy
import numpy as np
import sympy
from sympy.core.symbol import symbols
import scipy.linalg as sciline

x1,x2,x3 = symbols('x1,x2,x3')

def GaussianElimination(matrix:list, column, pivot_mod=0):
    """高斯消元法解线性方程组(仅考虑方阵形式),
        pivot_mod用于选择主元模式,
        默认0,0-不选择 1-部分主元 2-全主元"""
    matrix = np.array(matrix,dtype=float)
    column = np.array(column,dtype=float)
    length = len(matrix)
    for i in range(length):
        if pivot_mod != 0: 
            if pivot_mod == 1:
                max_index = i
                for item in range(i+1,length):
                    if abs(matrix[item][i]) > abs(matrix[max_index][i]): 
                        max_index = item
                if max_index != i: # 交换
                    pos = copy.deepcopy(matrix[max_index][:])
                    matrix[max_index][:] = matrix[i][:]
                    matrix[i][:] = pos
                    pos = copy.deepcopy(column[max_index][0])
                    column[max_index][0] = column[i][0]
                    column[i][0] = pos
            elif pivot_mod == 2: # 平衡主元只会进行一次
                if i == 0 :
                    max_scale = matrix[0][0]/max([abs(copy.deepcopy(matrix[0][j])) for j in range(length)])
                    max_index = 0
                    for k in range(1,length):
                        max_abs_one = max([abs(copy.deepcopy(matrix[0][j])) for j in range(length)])
                        if matrix[k][0]/max_abs_one > max_scale:
                            max_scale = matrix[k][0]/max_abs_one
                            max_index = k
                    if max_index != i: # 交换
                        pos = copy.deepcopy(matrix[max_index][:])
                        matrix[max_index][:] = matrix[i][:]
                        matrix[i][:] = pos
                        pos = copy.deepcopy(column[max_index][0])
                        column[max_index][0] = column[i][0]
                        column[i][0] = pos
                pass
        # 这里是用于一般的交换
        if matrix[i][i] == 0 :
            flag = 1
            for item in range(i+1,length):
                if matrix[item][i] != 0:
                    pos = copy.deepcopy(matrix[item][:])
                    matrix[item][:] = matrix[i][:]
                    matrix[i][:] = pos
                    pos = copy.deepcopy(column[item][0])
                    column[item][0] = column[i][0]
                    column[i][0] = pos
                    flag = 0
                    break
            if flag :
                return f"matrix is rank is less than {length}"

        pivot = matrix[i][i]   # 在这里获取主元
        j = i+1
        while j < length:
            times = float(matrix[j][i]/pivot)
            a = [matrix[j][k]-(matrix[i][k])*(times) for k in range(length)]
            matrix[j][:] = np.array(a,dtype=float)
            column[j][0] = column[j][0] - column[i][0] * times
            j += 1
    for i in reversed(range(length)):
        pivot = matrix[i][i]
        j = i-1
        while j > -1:
            times = float(matrix[j][i]/pivot)
            column[j][0] = column[j][0] - column[i][0] * times
            j -= 1
    return [[column[i][0] / matrix[i][i]] for i in range(length)]

if __name__ == "__main__":
    matrix = [[3.03,-12.1,14],\
              [-3.03,12.1,-7],\
              [6.11,-14.2,21]]
    column_vextor = [[-119],[120],[-139]]

    result_no_pivot = GaussianElimination(matrix=copy.deepcopy(matrix), column=copy.deepcopy(column_vextor), pivot_mod=0)
    print("直接计算:",result_no_pivot) # 默认
    result_partial_pivot = GaussianElimination(matrix=copy.deepcopy(matrix), column=copy.deepcopy(column_vextor), pivot_mod=1)
    print("部分主元:",result_partial_pivot) # 模式一
    result_scaled_partial_pivot = GaussianElimination(matrix=copy.deepcopy(matrix), column=copy.deepcopy(column_vextor), pivot_mod=2)
    print("平衡的部分主元:",result_scaled_partial_pivot) # 模式二
    result_scipy = sciline.solve(np.array(matrix),np.array(column_vextor))
    print("使用scipy库:",result_scipy)  # 使用scipy库

    system = copy.deepcopy(matrix)
    for i in range(len(matrix)):
        system[i].extend(column_vextor[i])
    result_sympy = sympy.solve_linear_system(sympy.Matrix(system),x1,x2,x3)
    print("使用sympy库:",result_sympy) # 使用sympy库
    symbol_list = [x1,x2,x3]

    # 使用scipy库的结果作为基准，构建误差列表，由于sympy的符号特性，不能使用sympy
    if type(result_no_pivot) != str : # 有解才有计算误差的意义！！！
        deviation_list = []
        deviation_list.append(sum([(result_no_pivot[i][0] - result_scipy[i][0])**2 for i in range(len(matrix))]))
        deviation_list.append(sum([(result_partial_pivot[i][0] - result_scipy[i][0])**2 for i in range(len(matrix))]))
        deviation_list.append(sum([(result_scaled_partial_pivot[i][0] - result_scipy[i][0])**2 for i in range(len(matrix))]))
        deviation_list.append(sum([(result_sympy[symbol_list[i]] - result_scipy[i][0])**2 for i in range(len(matrix))]))
        # 在程序运行的过程中，各种计算方法的误差大小差别其实已经是在小数点后十三位了，而且非常接近0，
        # 如果使用np.sart()开方的话会使得浮点数溢出异常，python因为np.sqrt()方法报错，因此就保留平方不做开方运算
        # deviation_list = [np.sqrt(i) for i in deviation_list)]
        print("误差列表:",deviation_list)
