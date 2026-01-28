import scipy.io
import pickle
        
def save_obj(obj,name):
    with open(name + '.pkl','wb') as f:
        pickle.dump(obj,f)

A = scipy.io.loadmat('./data/.../Dataset_nmc.mat') # data fron offical
Data = A['Train_Set']
Data_cell = int(Data['Cell'][0][300][0][0])
Data_history = Data['History'][0][0]
Bat = dict()
lbl = []
cell_num = [ x for x in range(1,48)]
for i in cell_num:
    lbl_cell = []
    for j in range(Data.shape[1]): # 
        if int(Data['Cell'][0][j][0][0]) == i:
            lbl_cell.append(Data['History'][0][j][0][-1])
    Bat.update({'Cell' + str(i):{'lbl':lbl_cell}})
save_obj(Bat,'D:/Edge下载文件夹/nmc_loader')