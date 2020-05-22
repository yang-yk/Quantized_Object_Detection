import h5py
import numpy as np

train_path_x = '/home/yyk17/yangyk/TianjiChip/quan_final/Quan_test0310/bunny/train_aps_x.mat'
train_path_y = '/home/yyk17/yangyk/TianjiChip/quan_final/Quan_test0310/bunny/train_aps_y.mat'

test_path_x = '/home/yyk17/yangyk/TianjiChip/quan_final/Quan_test0310/bunny/test_aps_x.mat'
test_path_y = '/home/yyk17/yangyk/TianjiChip/quan_final/Quan_test0310/bunny/test_aps_y.mat'

train_data_x, train_data_y = h5py.File(train_path_x)['train_aps_x'], h5py.File(train_path_y)['train_aps_y']
test_data_x, test_data_y = h5py.File(test_path_x)['test_aps_x'], h5py.File(test_path_y)['test_aps_y']

train_data_x = np.array(train_data_x)
train_data_y = np.array(train_data_y)

test_data_x = np.array(test_data_x)
test_data_y = np.array(test_data_y)

data_path='/home/yyk17/yangyk/TianjiChip/quan_final/Quan_test0310/bunny/'

np.save(data_path+'train_data_x.npy',train_data_x)
np.save(data_path+'train_data_y.npy',train_data_y)
np.save(data_path+'test_data_x.npy',test_data_x)
np.save(data_path+'test_data_y.npy',test_data_y)



print('over')