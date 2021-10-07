import numpy as np

def calculate_mean_and_std(row_data:list,args):
    res=[]
    for i in row_data:
        res.append(i[args[0]][args[1]])
    res=np.array(res)     
    return np.mean(res), np.std(res)

def calculate_similarity(row_data:list,countNums:dict):
    for key,values in countNums.items():
        print(f'{key} area corresponding is')
        for value in values:
            print(calculate_mean_and_std(row_data,value))
        print('-------------------------')   

