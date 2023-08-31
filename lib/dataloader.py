import torch
import numpy as np
import torch.utils.data
import pandas as pd


class StandardScaler():
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
        
    def transform(self, data):
        return (data - self.mean) / self.std
    
    def inverse_transform(self, data):
        return (data * self.std) + self.mean
    
    
class MinMaxScaler():
    def __init__(self,max_,min_):
        self.max_ = max_
        self.min_ = min_
    def transform(self, data):
        return (data - self.min_) / (self.max_ - self.min_)
    def inverse_transform(self, data):
        return data * (self.max_ - self.min_) + self.min_
    
'''
 If somehow your training does not explode then 
 the first few stages of the training will still be a waste as 
 the first thing the network will learn is 
 to scale and shift the output values into roughly the desired range. 
 
 consider normalization as the process of making 
 the "units" of all the input features equal.
 
 Having big discontinuities in the data space, 
 or large clusters of separated data which represent the same thing, 
 is going to make the learning task much more difficult.
'''        
files = {
    'pemsd3': ['PEMS03/PEMS03.npz', 'PEMS03/PEMS03.csv'],
    'pemsd4': ['PEMS04/PEMS04.npz', 'PEMS04/PEMS04.csv'],
    'pemsd7': ['PEMS07/PEMS07.npz', 'PEMS07/PEMS07.csv'],
    'pemsd8': ['PEMS08/PEMS08.npz', 'PEMS08/PEMS08.csv'],
    'case':['CASE/CASE.npz'],
    'metrla':['METRLA/METRLA.h5', 'METRLA/adj_mx,metr.pkl'],
    'pemsbay': ['PEMSBAY/PEMSBAY.h5', 'PEMSBAY/adj_mx_bay.pkl'],
    'bj500':['BJ500/BJ500.h5','BJ500/adj_mx_bj500.txt'],
    'solar_energy':['DATASET-SINGLE-STEP/solar_energy/solar_energy.txt.gz'],
    'electricity':['DATASET-SINGLE-STEP/electricity/electricity.txt.gz'],
    'exchange_rate':['DATASET-SINGLE-STEP/exchange_rate/exchange_rate.txt.gz'],
    'traffic':['DATASET-SINGLE-STEP/traffic/traffic.txt.gz']
#     'pemsD7M': ['PeMSD7M/PeMSD7M.npz', 'PeMSD7M/distance.csv'],
#     'pemsD7L': ['PeMSD7L/PeMSD7L.npz', 'PeMSD7L/distance.csv']
}

class DataLoader(object):
    
    def __init__(self, val_r, test_r, filename, filepath, period_in, period_out, horizon):
        
        self.val_r = val_r
        self.test_r = test_r
        self.train_r = 1 - val_r - test_r
        self.filename = filename.lower()
        self.file = files[self.filename]
        self.filepath = filepath
        self.period_in = period_in
        self.period_out = period_out
        self.h = horizon
        period_ele = 12*12
        self.period_ele = period_ele
        self.Means = []
        self.Stds = []
      
        
        if self.filename in ('pemsd8','pemsd7','pemsd4','pemsd3'):
            self.data = np.load(self.filepath + self.file[0])['data']#read npz
            time_stamp = np.zeros(period_ele)
            for i in range(period_ele):
                time_stamp[i] = i/period_ele #走dim = 2
                #time_stamp[i] = i #走one_hot
            len_ofD = self.data.shape[0]
            
            time_stamps = np.tile(time_stamp,(int(len_ofD // period_ele),))
            
            time_stamps = np.tile(time_stamps,(1,self.data.shape[1],1)).transpose(2,1,0)
            
            self.data = np.concatenate((self.data, np.round(time_stamps,decimals=4)), axis = -1)
            self.Te, self.Nu, self.Fe = self.data.shape
            print('Is there nan(s)?',np.where(np.isnan(self.data)))
            print('Shape of data',self.data.shape)
            
                    
        elif self.filename in ('metrla','pemsbay','bj500'):
            df = pd.read_hdf(self.filepath + self.file[0])# read h5 
            num_samples, num_nodes = df.shape
            self.data = np.expand_dims(df.values, axis=-1)# h5 转np
            
            time_stamp = np.zeros(period_ele)
            for i in range(period_ele):
                time_stamp[i] = i/period_ele
                #time_stamp[i] = i
            len_ofD = self.data.shape[0]
            if len_ofD % period_ele == 0:
                time_stamps = np.tile(time_stamp,(int(len_ofD // period_ele),))
            else:
                time_stamps = np.tile(time_stamp,(int(len_ofD // period_ele) + 1,))
                time_stamps = time_stamps[:len_ofD]
            time_stamps = np.tile(time_stamps,(1,self.data.shape[1],1)).transpose(2,1,0)
            
            self.data = np.concatenate((self.data, np.round(time_stamps,decimals=4)), axis = -1)
            self.Te, self.Nu, self.Fe = self.data.shape
            print('Is there nan(s)?',np.where(np.isnan(self.data)))
            print('Shape of data',self.data.shape)
            self.Te, self.Nu, self.Fe = self.data.shape
        else:
            self.data = np.loadtxt(self.filepath + self.file[0], delimiter=',')
            num_samples, num_nodes = self.data.shape
            self.data = np.expand_dims(self.data,axis = -1)
            self.Te, self.Nu, self.Fe = self.data.shape
            print('Is there nan(s)?',np.where(np.isnan(self.data)))
            print('Shape of data',self.data.shape)
        self.Te, self.Nu, self.Fe = self.data.shape
        ####################################   
        
    
    def _batchify2(self, in_tensor, idx_set):
        idx = range(self.period_in + self.period_out + self.h - 1, len(idx_set))
        n = len(idx)
        X = np.zeros((n, self.period_in, self.Nu, self.Fe))
        Y = np.zeros((n, self.period_out, self.Nu, self.Fe))
        for i in range(n):
            end = idx[i]
            start = end - (self.period_in + self.period_out + self.h - 1)
            
            X[i,:,:,:] = in_tensor[start:start+self.period_in,:,:]
            Y[i,:,:,:] = in_tensor[start+self.period_in+self.h:end+1,:,:]
        return [X,Y]
    
    def create_raw_matrix2(self):
        train_set = range(0, int(self.train_r * self.Te))
        valid_set = range(int(self.train_r * self.Te),int((self.val_r + self.train_r)* self.Te))
        test_set = range(int((self.val_r + self.train_r) * self.Te), self.Te)
        Dtrain = self.data[train_set]
        Dvalid = self.data[valid_set]
        Dtest = self.data[test_set]
        D1 = self._batchify2(Dtrain,train_set)
        D2 = self._batchify2(Dvalid,valid_set)
        D3 = self._batchify2(Dtest,test_set)
        events = [D1,D2,D3]
        for dim in range(self.Fe - 1):
            Mean = D1[0][...,dim].mean().astype('float32')
            Std = D1[0][...,dim].std().astype('float32')
            self.Means.append(Mean)
            self.Stds.append(Std)
            #scaler = StandardScaler(mean = Mean, std = Std)
            #for D in events:
                #D[0][...,dim] = scaler.transform(D[0][...,dim])
                #D[1][...,dim] = scaler.transform(D[1][...,dim])
        #tsave = np.sort(np.unique(np.concatenate((D1[0][...,-1],D1[1][...,-1]))))
        #t2id = {t: tid for tid, t in enumerate(tsave)}
        print('Train shape', events[0][0].shape,events[0][1].shape)
        print('Val shape', events[1][0].shape,events[1][1].shape)    
        print('Test shape', events[2][0].shape,events[2][1].shape)    
        #print('Header',events[0][0][0,1,0,:])
        #print('tsave',tsave)
        return events, self.Means, self.Stds
    

    
class DataLoader_(object):
    def __init__(self, xs, ys, batch_size, pad_with_last_sample=True):
        """
        :param xs:
        :param ys:
        :param batch_size:
        :param pad_with_last_sample: pad with the last sample to make number of samples divisible to batch_size.
        """
        self.batch_size = batch_size
        self.current_ind = 0
        if pad_with_last_sample:
            num_padding = (batch_size - (len(xs) % batch_size)) % batch_size
            x_padding = xs[-1:].repeat(num_padding,*len(xs.shape[1:])*(1,))
            y_padding = ys[-1:].repeat(num_padding,*len(ys.shape[1:])*(1,))
            xs = torch.cat([xs, x_padding], 0)
            ys = torch.cat([ys, y_padding], 0)
        self.size = len(xs)
        self.num_batch = int(self.size // self.batch_size)
        self.permutation = None
        self.xs = xs
        self.ys = ys

    def shuffle(self):
        self.permutation = np.random.permutation(self.size)
        xs, ys = self.xs[self.permutation], self.ys[self.permutation]
        self.xs = xs
        self.ys = ys

    def get_iterator(self):
        self.current_ind = 0
        def _wrapper():
            while self.current_ind < self.num_batch:
                start_ind = self.batch_size * self.current_ind
                end_ind = min(self.size, self.batch_size * (self.current_ind + 1))
                x_i = self.xs[start_ind: end_ind, ...]
                y_i = self.ys[start_ind: end_ind, ...]
                yield (x_i, y_i)
                self.current_ind += 1

        return _wrapper()    

def load_dataset(val_r, test_r, filename, filepath, 
                 batch_size, device, period_in, period_out , horizon = 0, valid_batch_size=None, test_batch_size=None):
   
    DL = DataLoader(val_r, test_r, filename, filepath, period_in , period_out, horizon)
    data = {}
    

    events, Ms, Ss = DL.create_raw_matrix2()
    for i in range(len(events)):
            for j in range(len(events[i])):
                events[i][j] = torch.from_numpy(events[i][j].astype(np.float32)).to(device)
#         node_num = events[0][0].shape[2]#S,s,Node_num,Fe
#         #若除不尽则加node维度
#         x = node_num % num_splits
#         if x != 0:
#             add_dim = int(num_splits - x)
    print('Shape of the first sequence:',events[0][0].shape,)
    print('Means: ',Ms,'\n','Stds: ',Ss)
    Mean = torch.tensor(Ms).to(device)
    Std = torch.tensor(Ss).to(device)
    data['scaler'] = StandardScaler(mean = Mean[0], std = Std[0])
    data['Node_num'] = events[0][0].shape[2]
    data['train_loader'] = DataLoader_(*events[0],batch_size)
    data['val_loader'] = DataLoader_(*events[1],batch_size)
    data['test_loader'] = DataLoader_(*events[2],batch_size)
    
   
    return data

# def load_dataset(val_r, test_r, filename, filepath, 
#                  batch_size, device, period_in, period_out, horizon, valid_batch_size=None, test_batch_size=None, single=False):
   
#     DL = DataLoader(val_r, test_r, filename, filepath, period_in, period_out, device, single)
#     data = {}
    
#     x_tra, y_tra, x_val, y_val, x_test, y_test, Mean, Std = DL.create_raw_matrix()
    
#     scaler = StandardScaler(Mean,Std)
#     print('Shape of the first sequence:',x_tra.shape,)
#     print('Means: ',Mean,'\n','Stds: ',Std)
#     data['Node_num'] = x_tra.shape[2]
#     data['train_loader'] = DataLoader_(x_tra,y_tra,batch_size)
#     data['val_loader'] = DataLoader_(x_val,y_val,batch_size)
#     data['test_loader'] = DataLoader_(x_test,y_test,batch_size)
#     data['scaler'] = scaler
#     return data

