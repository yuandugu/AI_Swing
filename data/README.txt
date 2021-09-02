服务器数据地址

二阶模型数据地址：
/public/home/spy2018/swing/result/IEEE/case'N'/omega=='omega'/change_'type'_node_long/'data_number'/
高阶模型数据地址：
/public/home/spy2018/swing/result/TDS、case'N'/case'N'/perturbation/
其中各变量定义见 utils/load_data.py

二阶模型计算程序

使用mpi4py库进行并行运算，使用方法较为简单可自行谷歌

高阶模型程序

需要下载Power System Toolbox，主程序为 s_simu.m，数据文件为 datane.m