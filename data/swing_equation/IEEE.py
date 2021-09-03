from mpi4py import MPI
import math
import numpy as np
from scipy.integrate import solve_ivp
from scipy.special import comb, perm
import xlrd
import time
import random
import pandas as pd
import timeit
import operator
import h5py

#####################  parameters  ####################

N = 39                           # number of node
omega_s = 100 * math.pi          # synchronous angular frequency
baseMVA = 10**8                  # power reference value
M = 50000                        # mass moments of inertia                  
alpha = 0.6                      # damping
theta = math.pi                  # range of theta_0
omega = 20                       # range of omega_0
step = 0.05                      # time step to solve ODE
max_t = 120                      # maximum time to sove ODE
t = np.arange(0, max_t, step)    # time stream to solve ODE
data_number = 1000               # samping number
interval = False
if interval == True:
    cut_out_num = 50                # collect data number, 100 for 14, 50 for 39
else:
    cut_out_num = 100

def dmove(t, y, sets):
    """
    定义ODE
    """
    X = np.zeros((N * 2))
    for i in range(N):
        X[i] = y[i + N]
        a = 0
        for j in range(N):
            a += sets[i + 1, j]/16 * math.sin(y[j] - y[i])
        X[i + N] = -alpha * y[i + N] + sets[0, i]/16 + a
    return X

def load_para():
    parameter = xlrd.open_workbook('/parameter/parameter%s.xlsx' %(N))
    # 功率矩阵
    P_sheet1 = parameter.sheet_by_index(0)
    nrows = P_sheet1.nrows
    ncols = P_sheet1.ncols
    P = np.zeros((N))
    for i in range(nrows):
        for j in range(ncols):
            P[i] = P_sheet1.cell_value(i, j)
    P = P * baseMVA
    P = [i - np.sum(P)/N for i in P]  # 功率补偿
    P = np.array([i/(M*omega_s) for i in P])
    # 导纳矩阵
    Y_sheet1 = parameter.sheet_by_index(1)
    nrows = Y_sheet1.nrows
    ncols = Y_sheet1.ncols
    Y = np.zeros((N, N))
    for i in range(nrows):
        for j in range(ncols):
            Y[i, j] = Y_sheet1.cell_value(i, j)
    Y = np.array([i*baseMVA/(M*omega_s) for i in Y])
    # 参数合并
    PY = np.vstack((P, Y))
    # 初始条件
    theta_sheet1 = parameter.sheet_by_index(2)
    nrows = theta_sheet1.nrows
    ncols = theta_sheet1.ncols
    initial = np.zeros((N * 2))
    for i in range(nrows):
        for j in range(ncols):
            initial[i] = theta_sheet1.cell_value(i, j)
    initial = [i / 180 * math.pi for i in initial]  # 转换为弧度制
    print('原始数据导入完毕')
    return PY, initial

def generate_uniform_init_array(Initial, init_num, node_num):
    """
    产生多组单个节点服从均匀分布的随机初始条件
    """
    np.random.seed(node_num*570)
    init_array = np.random.rand(2, init_num)
    init_array -= 0.5*np.ones((2, init_num))
    init_array[0, :] *= 2 * theta
    init_array[0, :] += Initial[node_num - 1] * np.ones((init_num))
    init_array[1, :] *= 2 * omega
    return init_array

def solve_one_ODE_updated(i):
    """
    parallel function
    """
    if N == 14:
        length = 4000
    elif N == 39:
        length = 1000
    names = locals()
    a = np.array([-0.24219997, -0.16992011, -0.21896319, -0.22769395, -0.20274313, -0.18877805,
                -0.23072831, -0.24088105, -0.25411382, -0.14792818, -0.16214242, -0.16401846,
                -0.16169114, -0.1933527,  -0.20324505, -0.17720979, -0.19711253, -0.21354782,
                -0.08796499, -0.11204258, -0.13237097, -0.04721098, -0.05117464, -0.1747437,
                -0.14210796, -0.16254737, -0.20094919, -0.09408921, -0.04086045, -0.12485783,
                -0.021106,   -0.01778558,  0.00184892, -0.02056255,  0.04571267,  0.10145837,
                -0.01671788,  0.08897803, -0.26130884, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                0., 0., 0., 0., 0., 0., 0., 0., 0.]) # IEEE-39的同步状态
    names['init_'+str(i)] = generate_uniform_init_array(Initial=a, init_num=data_number, node_num=i+1) # 第i+1个节点的初始条件
    S = []
    data_theta = np.zeros((data_number, cut_out_num * N))
    data_omega = np.zeros((data_number, cut_out_num * N))
    for j in range(data_number):
        init = a
        init[i] = names['init_'+str(i)][0, j]
        init[i+N] = names['init_'+str(i)][1, j]
        names['result' + str(i) + str(j)] = solve_ivp(fun=lambda t, y: dmove(t, y, PY), t_span=(0.0, max_t),  y0=init, method='RK45', t_eval=t)
        for num in range(N):
            if interval == True:
                data_theta[j, num*cut_out_num:(num*cut_out_num+cut_out_num)] = names['result' + str(i) + str(j)].y[num, 0:4*cut_out_num-3:4]
                data_omega[j, num*cut_out_num:(num*cut_out_num+cut_out_num)] = names['result' + str(i) + str(j)].y[num+N, 0:4*cut_out_num-3:4]
            else:
                data_theta[j, num*cut_out_num:(num*cut_out_num+cut_out_num)] = names['result' + str(i) + str(j)].y[num, 0:cut_out_num]
                data_omega[j, num*cut_out_num:(num*cut_out_num+cut_out_num)] = names['result' + str(i) + str(j)].y[num+N, 0:cut_out_num]       
        if(np.amax(abs(names['result' + str(i) + str(j)].y[N:, -1])) <= 0.2):
            S.append(0)  # 收敛
            print(0)
        else:
            S.append(1)  # 不收敛
            print(1)

        del names['result' + str(i) + str(j)], init
        print('第(%s,%s)个ODE计算结束' % (i+1, j+1))
    if interval == True:
        f = h5py.File('/one/%s.h5' % (i+1), 'w')
    else:
        f = h5py.File('/one/%s.h5' % (i+1), 'w')        
    f.create_dataset('data_theta', data=data_theta)
    f.create_dataset('data_omega', data=data_omega)
    f.create_dataset('Y', data=np.array(S))
    f.close()

def bigjobMPI_one_updated():
    """
    calculate change_two_node data
    """
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()
    
    numjobs = N

    job_content = []  # the collection of parameters [i,j]
    for i_cur in range(N):
        job_content.append(i_cur)

    # arrange the works and jobs
    if rank == 0:
        # this is head worker
        # jobs are arranged by this worker
        job_all_idx = list(range(numjobs))
        random.shuffle(job_all_idx)
        # shuffle the job index to make all workers equal
        # for unbalanced jobs
    else:
        job_all_idx = None
    
    job_all_idx = comm.bcast(job_all_idx, root=0)
    
    njob_per_worker, res = divmod(numjobs, size)
    # the number of jobs should be a multiple of the NumProcess[MPI]
    if rank < res:
        this_worker_job = [job_all_idx[x] for x in range(rank*(njob_per_worker+1), (rank + 1)*(njob_per_worker+1))]
    elif rank >= res:
        this_worker_job = [job_all_idx[x] for x in range(rank*njob_per_worker + res, (rank + 1)*njob_per_worker + res)]

    # map the index to parameterset [eps,anis]
    work_content = [job_content[x] for x in this_worker_job]

    for a_piece_of_work in work_content:
        print('核心数为:%s' %(rank))
        solve_one_ODE_updated(a_piece_of_work)

if __name__=="__main__": 
    
    PY, initial = load_para()
    bigjobMPI_one_updated()
