import numpy as np
import pandas as pd
import datetime, os, time
import multiprocessing as mp
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler  # 新增：用于因子标准化
import xgboost as xgb

class OlsModel:
    def __init__(self):
        # --------------------------
        # 1. Kaggle数据路径调整
        # --------------------------
        self.train_data_path = "train_data"  # Kaggle训练数据文件夹路径（需替换为实际路径）
        self.submission_id_path = "submission_id.csv"  # 提交ID文件路径
        self.start_datetime = datetime.datetime(2021, 3, 1, 0, 0, 0)
        self.all_train_data = []  # 新增：用于全局分数统计
    
    def get_all_symbol_list(self):
        try:
            parquet_name_list = os.listdir(self.train_data_path)
            symbol_list = [parquet_name.split(".")[0] for parquet_name in parquet_name_list if parquet_name.endswith(".parquet")]  # 过滤非parquet文件
            return symbol_list
        except Exception as e:
            print(f"get_all_symbol_list error: {e}")
            return []
    
    def get_single_symbol_kline_data(self, symbol):
        try:
            df = pd.read_parquet(f"{self.train_data_path}/{symbol}.parquet")
            df = df.set_index("timestamp")
            df = df.astype(np.float64)
            # 处理volume为0的情况（避免除零错误）
            df['vwap'] = np.where(df['volume'] == 0, np.nan, df['amount'] / df['volume'])
            df['vwap'] = df['vwap'].replace([np.inf, -np.inf], np.nan).ffill()
            df.columns.name = symbol  # 标记列所属符号，便于后续合并
            return df
        except Exception as e:
            print(f"get_single_symbol_kline_data error (symbol: {symbol}): {e}")  # 补充符号名，便于调试
            return pd.DataFrame()
    
    def get_all_symbol_kline(self):
        t0 = datetime.datetime.now()
        print(f"[get_all_symbol_kline] 开始读取币种数据，共{len(self.get_all_symbol_list())}个币种...")
        all_symbol_list = self.get_all_symbol_list()
        if not all_symbol_list:
            print("No symbols found in train_data_path.")
            return [], [], [], [], [], [], [], []
        df_list = []
        for idx, symbol in enumerate(all_symbol_list):
            if (idx+1) % 100 == 0 or idx == 0 or idx == len(all_symbol_list)-1:
                print(f"[get_all_symbol_kline] 进度: {idx+1}/{len(all_symbol_list)}")
            df = self.get_single_symbol_kline_data(symbol)
            df_list.append(df)
        print(f"[get_all_symbol_kline] 数据拼接完成, 用时: {datetime.datetime.now() - t0}")
        # 分别拼接每个字段，列名为symbol
        df_open_price = pd.concat([df['open_price'] for df in df_list], axis=1)
        df_high_price = pd.concat([df['high_price'] for df in df_list], axis=1)
        df_low_price = pd.concat([df['low_price'] for df in df_list], axis=1)
        df_close_price = pd.concat([df['close_price'] for df in df_list], axis=1)
        df_vwap = pd.concat([df['vwap'] for df in df_list], axis=1)
        df_amount = pd.concat([df['amount'] for df in df_list], axis=1)
        df_open_price = df_open_price.sort_index(ascending=True)
        time_arr = pd.to_datetime(df_open_price.index, unit='ms').values
        open_price_arr = df_open_price.values.astype(float)
        high_price_arr = df_high_price.sort_index(ascending=True).values.astype(float)
        low_price_arr = df_low_price.sort_index(ascending=True).values.astype(float)
        close_price_arr = df_close_price.sort_index(ascending=True).values.astype(float)
        vwap_arr = df_vwap.sort_index(ascending=True).values.astype(float)
        amount_arr = df_amount.sort_index(ascending=True).values.astype(float)
        return all_symbol_list, time_arr, open_price_arr, high_price_arr, low_price_arr, close_price_arr, vwap_arr, amount_arr
    
    def weighted_spearmanr(self, y_true, y_pred):
        n = len(y_true)
        if n < 2:
            return 0.0  # 样本量过小时返回0，避免计算错误
        r_true = pd.Series(y_true).rank(ascending=False, method='average')
        r_pred = pd.Series(y_pred).rank(ascending=False, method='average')
        x = 2 * (r_true - 1) / (n - 1) - 1
        w = x **2  
        w_sum = w.sum()
        if w_sum == 0:
            return 0.0  # 避免权重和为0导致除零
        mu_true = (w * r_true).sum() / w_sum
        mu_pred = (w * r_pred).sum() / w_sum
        cov = (w * (r_true - mu_true) * (r_pred - mu_pred)).sum()
        var_true = (w * (r_true - mu_true)** 2).sum()
        var_pred = (w * (r_pred - mu_pred) **2).sum()
        if var_true == 0 or var_pred == 0:
            return 0.0  # 避免方差为0导致除零
        return cov / np.sqrt(var_true * var_pred)

    def train(self, df_target, df_factor1, df_factor2, df_factor3, batch_results=None, save_result=False):
        t0 = time.time()
        scaler = StandardScaler()
        factor1_long = df_factor1.stack()
        factor2_long = df_factor2.stack()
        factor3_long = df_factor3.stack()
        target_long = df_target.stack()
        factor1_long.name = 'factor1'
        factor2_long.name = 'factor2'
        factor3_long.name = 'factor3'
        target_long.name = 'target'
        data = pd.concat([factor1_long, factor2_long, factor3_long, target_long], axis=1).dropna()
        if data.empty:
            return None
        X = data[['factor1', 'factor2', 'factor3']]
        y = data['target']
        model = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X, y)
        data['y_pred'] = model.predict(X)
        # 只保留特征重要性和分数输出
        # importance = model.feature_importances_
        # print(f"[train] 特征重要性: {dict(zip(X.columns, importance))}")
        # print(f"[train] Weighted Spearman: {self.weighted_spearmanr(data['target'], data['y_pred']):.4f} | 批耗时: {time.time() - t0:.1f}s")
        # self.all_train_data.append(data[['target', 'y_pred']])
        df_submit = data.reset_index()
        df_submit = df_submit.rename(columns={'level_0': 'datetime', 'level_1': 'symbol'})
        df_submit = df_submit[['datetime', 'symbol', 'y_pred']]
        df_submit['datetime'] = pd.to_datetime(df_submit['datetime'])
        df_submit["id"] = df_submit["datetime"].astype(str) + "_" + df_submit["symbol"]
        df_submit = df_submit[['id', 'y_pred']].rename(columns={'y_pred': 'predict_return'})
        if batch_results is not None:
            batch_results.append(df_submit)
        if save_result:
            try:
                df_submission_id = pd.read_csv(self.submission_id_path)
                id_list = df_submission_id["id"].tolist()
                df_submit_competion = df_submit[df_submit['id'].isin(id_list)].copy()
                missing_elements = list(set(id_list) - set(df_submit_competion['id']))
                new_rows = pd.DataFrame({'id': missing_elements, 'predict_return': [0] * len(missing_elements)})
                df_submit_competion = pd.concat([df_submit_competion, new_rows], ignore_index=True)
                df_submit_competion = df_submit_competion.set_index('id').loc[id_list].reset_index()
                df_submit_competion.to_csv("submit.csv", index=False)
                print("[train] Submit file saved as 'submit.csv'.")
            except Exception as e:
                print(f"Error saving submit file: {e}")
        return df_submit

    def run(self):
        print("[run] 开始读取K线数据...")
        (all_symbol_list, time_arr, open_price_arr, high_price_arr, low_price_arr, 
         close_price_arr, vwap_arr, amount_arr) = self.get_all_symbol_kline()
        if not all_symbol_list:
            print("[run] No data loaded. Exiting run().")
            return
        df_vwap = pd.DataFrame(vwap_arr, columns=all_symbol_list, index=time_arr)
        df_amount = pd.DataFrame(amount_arr, columns=all_symbol_list, index=time_arr)
        print("[run] 数据加载完毕，开始训练...")
        windows_1d = 4 * 24 * 1
        windows_7d = 4 * 24 * 7
        batch_size = 10
        time_index = pd.to_datetime(df_vwap.index)
        months = sorted(set([(d.year, d.month) for d in time_index]))
        batch_results = []
        for i in range(0, len(all_symbol_list), batch_size):
            batch_symbols = all_symbol_list[i:i+batch_size]
            print(f"[run] 批次: {i//batch_size+1}/{len(all_symbol_list)//batch_size+1}")
            df_vwap_batch = df_vwap[batch_symbols]
            df_amount_batch = df_amount[batch_symbols]
            for year, month in months:
                mask = (time_index.year == year) & (time_index.month == month)
                if not mask.any():
                    continue
                df_vwap_month = df_vwap_batch.loc[mask]
                df_amount_month = df_amount_batch.loc[mask]
                if df_vwap_month.empty or df_amount_month.empty:
                    continue
                df_24hour_rtn = df_vwap_month / df_vwap_month.shift(windows_1d) - 1
                df_15min_rtn = df_vwap_month / df_vwap_month.shift(1) - 1
                df_7d_volatility = df_15min_rtn.rolling(windows_7d).std(ddof=1)
                df_7d_momentum = df_vwap_month / df_vwap_month.shift(windows_7d) - 1
                df_amount_sum = df_amount_month.rolling(windows_7d).sum()
                self.train(
                    df_target=df_24hour_rtn.shift(-windows_1d),
                    df_factor1=df_7d_volatility,
                    df_factor2=df_7d_momentum,
                    df_factor3=df_amount_sum,
                    batch_results=batch_results,
                    save_result=False
                )
        if batch_results:
            df_submit_all = pd.concat(batch_results, ignore_index=True)
            df_submit_all = df_submit_all.drop_duplicates(subset=['id'], keep='last')
            print(f"[run] Total predictions: {len(df_submit_all)}")
            df_submit_all.to_csv("submit_all.csv", index=False)
            print("[run] All batch results saved as 'submit_all.csv'.")
            df_submission_id = pd.read_csv(self.submission_id_path)
            id_list = df_submission_id["id"].tolist()
            df_submit_competion = df_submit_all[df_submit_all['id'].isin(id_list)].copy()
            missing_elements = list(set(id_list) - set(df_submit_competion['id']))
            new_rows = pd.DataFrame({'id': missing_elements, 'predict_return': [0] * len(missing_elements)})
            df_submit_competion = pd.concat([df_submit_competion, new_rows], ignore_index=True)
            df_submit_competion = df_submit_competion.set_index('id').loc[id_list].reset_index()
            print(f"[run] Final submission shape: {df_submit_competion.shape} (expected: {len(id_list)})")
            df_submit_competion.to_csv("submit.csv", index=False)
            print("[run] Submit file saved as 'submit.csv'.")
        else:
            print("[run] No batch results generated.")
        if self.all_train_data:
            all_data = pd.concat(self.all_train_data, ignore_index=True)
            score = self.weighted_spearmanr(all_data['target'], all_data['y_pred'])
            print(f'[run] 全局Weighted Spearman correlation coefficient: {score:.4f}')


if __name__ == '__main__':
    model = OlsModel()
    model.run()