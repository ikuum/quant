import numpy as np
import pandas as pd
import datetime
import os
import time
import pickle
import multiprocessing as mp
import gc
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression   # 你原类里引用了，可保留
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset

class OlsModel:
    def __init__(self):
        # the folder path for setting sequence data
        self.train_data_path = "autodl-tmp/kline_data/train_data"
        self.start_datetime = datetime.datetime(2021, 3, 1, 0, 0, 0)
    
    def get_all_symbol_list(self):
        parquet_name_list = os.listdir(self.train_data_path)
        # 只保留前100个
        parquet_name_list = parquet_name_list[:50]
        symbol_list = [p.split(".")[0] for p in parquet_name_list]
        return symbol_list
    
    def get_single_symbol_kline_data(self, symbol):
        try:
            # 读取指定币种的K线数据，并将其设置为DataFrame
            df = pd.read_parquet(f"{self.train_data_path}/{symbol}.parquet",engine='fastparquet')
            df = df.set_index("timestamp")
            df = df.astype(np.float64)
    
            # 将时间戳转换为datetime格式
            df.index = pd.to_datetime(df.index, unit='ms')
    
            # 计算VWAP（成交量加权平均价），处理无效值
            df['vwap'] = (df['amount'] / df['volume']).replace([np.inf, -np.inf], np.nan).ffill()
    
            # 删除NaN值（处理缺失值）
            df = df.dropna()
    
            # 删除2021年3月1日之前的数据
            df = df[df.index >= self.start_datetime]
        except Exception as e:
            print(f"get_single_symbol_kline_data error: {e}")
            df = pd.DataFrame()
        return df        # ← 这里缩进到与 try 同级
     
    def get_all_symbol_kline(self):
        t0 = datetime.datetime.now()
         # create a process pool, using the number of available CPU cores minus 2, for parallel processing
        pool = mp.Pool(mp.cpu_count() - 2)
        # get a list of all currencies
        all_symbol_list = self.get_all_symbol_list()
         # the initialization list is used to store the results returned by each asynchronous read task
        df_list = []
        for i in range(len(all_symbol_list)):
            df_list.append(pool.apply_async(self.get_single_symbol_kline_data, (all_symbol_list[i], )))
        # the process pool is closed and no new tasks will be accepted
        pool.close()
        # wait for all asynchronous tasks to complete
        pool.join()
        # collect the opening price series of all asynchronous results and concatenate them into a DataFrame by columns, then sort the index in ascending order of time
        df_open_price = pd.concat([i.get()['open_price'] for i in df_list], axis=1).sort_index(ascending=True)
        # convert the time index (milliseconds) to a datetime type array
        time_arr = pd.to_datetime(pd.Series(df_open_price.index), unit = "ms").values
        # get the values from the opening price in the DataFrame and convert them into a NumPy array of float type
        open_price_arr = df_open_price.values.astype(float)
        # get the values from the highest price in the DataFrame and convert them into a NumPy array of float type
        high_price_arr = pd.concat([i.get()['high_price'] for i in df_list], axis=1).sort_index(ascending=True).values
        # get the values from the lowest price in the DataFrame and convert them into a NumPy array of float type
        low_price_arr = pd.concat([i.get()['low_price'] for i in df_list], axis=1).sort_index(ascending=True).values
        # get the values from the closing price in the DataFrame and convert them into a NumPy array of float type
        close_price_arr = pd.concat([i.get()['close_price'] for i in df_list], axis=1).sort_index(ascending=True).values
        # collect the volume-weighted average price series of all currencies and concatenate them into an array by columns
        vwap_arr = pd.concat([i.get()['vwap'] for i in df_list], axis=1).sort_index(ascending=True).values
        # collect the trading amount series of all currencies and concatenate them into an array by columns
        amount_arr = pd.concat([i.get()['amount'] for i in df_list], axis=1).sort_index(ascending=True).values
        print(f"finished get all symbols kline, time escaped {datetime.datetime.now() - t0}")
        return all_symbol_list, time_arr, open_price_arr, high_price_arr, low_price_arr, close_price_arr, vwap_arr, amount_arr
    
    def weighted_spearmanr(self, y_true, y_pred):
        """
        Calculate the weighted Spearman correlation coefficient according to the formula in the appendix:
        1) Rank y_true and y_pred in descending order (rank=1 means the maximum value)
        2) Normalize the rank indices to [-1, 1], then square to obtain the weight w_i
        3) Calculate the correlation coefficient using the weighted Pearson formula
        """
        # number of samples
        n = len(y_true)
        # rank the true values in descending order (average method for handling ties)
        r_true = pd.Series(y_true).rank(ascending=False, method='average')
        # rank the predicted values in descending order (average method for handling ties)
        r_pred = pd.Series(y_pred).rank(ascending=False, method='average')
        
        # normalize the index i = rank - 1, mapped to [-1, 1]
        x = 2 * (r_true - 1) / (n - 1) - 1
        # weight w_i (the weight factor for each sample)
        w = x ** 2  
        
        # weighted mean
        w_sum = w.sum()
        mu_true = (w * r_true).sum() / w_sum
        mu_pred = (w * r_pred).sum() / w_sum
        
        # calculate the weighted covariance
        cov = (w * (r_true - mu_true) * (r_pred - mu_pred)).sum()
        # calculate the weighted variance of the true value rankings
        var_true = (w * (r_true - mu_true)**2).sum()
        # calculate the weighted variance of the predicted value rankings
        var_pred = (w * (r_pred - mu_pred)**2).sum()
        
        # return the weighted Spearman correlation coefficient
        return cov / np.sqrt(var_true * var_pred)


 #修改区域
    def train(self, df_target, df_factor1, df_factor2, df_factor3):
        """
        一次性把所有数据预处理到内存，然后全 GPU 训练
        """
        look_back = 60
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.backends.cudnn.benchmark = True  # 加速卷积类模型，如 LSTM
        print("Using device:", device)

        # -------------------------------------------------
        # 1) 拉平长表
        # -------------------------------------------------
        factor1_long = df_factor1.stack().rename('factor1')
        factor2_long = df_factor2.stack().rename('factor2')
        factor3_long = df_factor3.stack().rename('factor3')
        target_long  = df_target.stack().rename('target')
        data = pd.concat([factor1_long, factor2_long, factor3_long, target_long], axis=1).dropna()

        # -------------------------------------------------
        # 2) 一次性预处理：rolling、标准化、序列化
        # -------------------------------------------------
        cache_file = "gpu_train_cache.pkl"
        if os.path.exists(cache_file):
            print("加载缓存数据...")
            with open(cache_file, "rb") as f:
                X_all, y_all, idx_all = pickle.load(f)
        else:
            print("首次运行，预处理数据...")
            scaler = StandardScaler()
            symbol_list = data.index.get_level_values(1).unique()
            X_all, y_all, idx_all = [], [], []

            for sym in tqdm(symbol_list, desc="预处理"):
                df_sym = data.xs(sym, level=1).sort_index()
                if len(df_sym) < look_back + 1:
                    continue

                # 提前做 rolling / pct_change
                df_sym['factor1'] = df_sym['factor1'].rolling(672).std()
                df_sym['factor2'] = df_sym['factor2'].pct_change(672)
                df_sym['factor3'] = df_sym['factor3'].rolling(672).sum()
                df_sym = df_sym.replace([np.inf, -np.inf], np.nan).dropna()
                if len(df_sym) < look_back + 1:
                    continue

                feats = scaler.fit_transform(df_sym[['factor1', 'factor2', 'factor3']])
                tgt   = df_sym['target'].values

                for i in range(look_back, len(feats)):
                    X_all.append(feats[i-look_back:i])
                    y_all.append(tgt[i])
                    idx_all.append((df_sym.index[i], sym))

            X_all = np.array(X_all, dtype=np.float32)
            y_all = np.array(y_all, dtype=np.float32)

            with open(cache_file, "wb") as f:
                pickle.dump((X_all, y_all, idx_all), f)

        # -------------------------------------------------
        # 3) 转到 GPU Tensor
        # -------------------------------------------------
        X = torch.tensor(X_all).to(device)
        y = torch.tensor(y_all).unsqueeze(1).to(device)
        dataset = TensorDataset(X, y)
        loader = DataLoader(
            dataset,
            batch_size=1024,
            shuffle=True,
            pin_memory=False,   # 可保留，提升主线程加载性能
            num_workers=0      # 避免子进程初始化 GPU
)
        # -------------------------------------------------
        # 4) 定义优化后的 LSTM 模型
        # -------------------------------------------------
        class LSTMRegressor(nn.Module):
            def __init__(self, input_dim=3, hidden_dim=512, num_layers=2, dropout=0.2):
                super().__init__()
                self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers,
                                    batch_first=True, dropout=dropout)
                self.fc = nn.Linear(hidden_dim, 1)

            def forward(self, x):
                out, _ = self.lstm(x)
                return self.fc(out[:, -1, :])

        model = LSTMRegressor().to(device)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        print("开始训练前释放显存缓存...")
        gc.collect()
        torch.cuda.empty_cache()

        # -------------------------------------------------
        # 5) GPU训练过程
        # -------------------------------------------------
        model.train()
        t0 = time.time()
        for epoch in range(20):  # 提升训练轮数
            epoch_loss = 0.0
            for xb, yb in tqdm(loader, desc=f"Epoch {epoch+1}"):
                optimizer.zero_grad()
                pred = model(xb)
                loss = criterion(pred, yb)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            print(f"Epoch {epoch+1}  loss={epoch_loss/len(loader):.6f}")
        print("训练总耗时:", time.time() - t0, "秒")


        
    

        # 6. 预测
        model.eval()
        y_pred_all = []
        
        # 不需要 shuffle 和 pin_memory
        pred_loader = DataLoader(TensorDataset(X), batch_size=2048)
        
        with torch.no_grad():
            for xb in tqdm(pred_loader, desc="Predicting"):
                xb = xb[0].to(device)
                pred = model(xb)
                y_pred_all.append(pred.cpu())
        
        y_pred = torch.cat(y_pred_all).numpy().flatten()

        pred_df = pd.DataFrame({
            'datetime': [t[0] for t in idx_all],
            'symbol':   [t[1] for t in idx_all],
            'predict_return': y_pred
        })

        # 7. 对齐 submission
        pred_df['id'] = pred_df['datetime'].astype(str) + '_' + pred_df['symbol']
        df_submission_id = pd.read_csv("submission_id.csv")
        df_submit = (pred_df.set_index('id')
                            .reindex(df_submission_id['id'])
                            .fillna(0)
                            .reset_index())
        df_submit = df_submit[['id', 'predict_return']]
        df_submit.to_csv("submit.csv", index=False)

        # 8. 可选：计算加权 Spearman
        true_vals = np.array(y_all)
        rho = self.weighted_spearmanr(true_vals, y_pred)
        print(f"Train-Weighted Spearman: {rho:.4f}")


    def run(self):
        all_symbol_list, time_arr, open_price_arr, high_price_arr, low_price_arr, \
            close_price_arr, vwap_arr, amount_arr = self.get_all_symbol_kline()

        df_vwap   = pd.DataFrame(vwap_arr,   columns=all_symbol_list, index=time_arr)
        df_amount = pd.DataFrame(amount_arr, columns=all_symbol_list, index=time_arr)

        windows_1d = 4 * 24 * 1
        windows_7d = 4 * 24 * 7

        df_24h_rtn   = df_vwap.pct_change(windows_1d)
        df_15min_rtn = df_vwap.pct_change()
        df_7d_vol    = df_15min_rtn.rolling(windows_7d).std()
        df_7d_mom    = df_vwap.pct_change(windows_7d)
        df_amount_sum= df_amount.rolling(windows_7d).sum()

        self.train(df_24h_rtn.shift(-windows_1d),
                   df_7d_vol,
                   df_7d_mom,
                   df_amount_sum)
    
    def run(self):
        # call the get_all_symbol_kline function to get the K-line data and event data for all currencies
        all_symbol_list, time_arr, open_price_arr, high_price_arr, low_price_arr, close_price_arr, vwap_arr, amount_arr = self.get_all_symbol_kline()
        # convert the vwap array into a DataFrame, with currencies as columns and time as the index (next line sets the index)
        df_vwap = pd.DataFrame(vwap_arr, columns=all_symbol_list, index=time_arr)
        # convert the amount array into a DataFrame, with currencies as columns and time as the index
        df_amount = pd.DataFrame(amount_arr, columns=all_symbol_list, index=time_arr)
        windows_1d = 4 * 24 * 1
        windows_7d = 4 * 24 * 7
        # calculate the return for the past 24 hours using rolling calculation
        df_24hour_rtn = df_vwap / df_vwap.shift(windows_1d) - 1
        # calculate the return for the past 15 minutes using rolling calculation
        df_15min_rtn = df_vwap / df_vwap.shift(1) - 1
        # calculate the first factor: 7-day volatility factor
        df_7d_volatility = df_15min_rtn.rolling(windows_7d).std(ddof=1)
        # calculate the second factor: 7-day momentum factor
        df_7d_momentum = df_vwap / df_vwap.shift(windows_7d) - 1
        # calculate the third factor: 7-day total volume factor
        df_amount_sum = df_amount.rolling(windows_7d).sum()
        # call the train method, using the lagged 7-day 24-hour return as the target value, and the three factors as inputs for model training
        self.train(df_24hour_rtn.shift(-windows_1d), df_7d_volatility, df_7d_momentum, df_amount_sum)   
        
        
        
if __name__ == '__main__':
    model = OlsModel()
    model.run()