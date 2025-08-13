from importlib import import_module

import numpy as np
import pandas as pd
import datetime, os, time
import multiprocessing as mp
from sklearn.linear_model import LinearRegression
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import datetime
import gc


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, output_size=1, dropout=0.2):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # 初始化隐藏状态和细胞状态
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # LSTM 前向传播
        out, _ = self.lstm(x, (h0, c0))

        # 取最后一个时间步的输出
        out = self.dropout(out[:, -1, :])
        out = self.fc(out)
        return out


class OlsModel:
    # 初始化，该类包含训练数据路径和开始日期两个实体
    def __init__(self):
        # 设置序列数据的文件夹路径
        self.train_data_path = "train_data"
        self.start_datetime = datetime.datetime(2021, 3, 1, 0, 0, 0)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'  # 自动选择GPU或CPU
        print(f"Using device: {self.device}")  # 打印使用的设备

    # 得到所有币种的符号（用来读取对应币种的数据文件）
    def get_all_symbol_list(self):
        # 获取训练数据目录中的所有文件名
        parquet_name_list = os.listdir(self.train_data_path)
        # 移除文件扩展名，只保留货币代码符号以生成货币代码列表
        symbol_list = [parquet_name.split(".")[0] for parquet_name in parquet_name_list]
        return symbol_list

    # 新增列如rsi、macd、atr、buy_ratio、vwap_deviation
    def compute_factors_pandas(self, df):
        """
        使用 pandas 和 numpy 计算技术指标因子，完全避免 PyTorch 和 CUDA。
        这可以防止在多进程数据加载时出现 CUDA 错误。
        """
        # 提取 numpy 数组以进行高效计算
        close = df['close_price'].values
        volume = df['volume'].values
        amount = df['amount'].values
        high = df['high_price'].values
        low = df['low_price'].values
        buy_volume = df['buy_volume'].values
        vwap = df['vwap'].values

        # ------------------ 计算相对强弱指数(RSI) ------------------
        # 计算价格变化
        delta = np.diff(close, prepend=close[0])  # 使用 np.diff，并用第一个价格填充开头
        gain = np.where(delta > 0, delta, 0.0)  # 正变化
        loss = np.where(delta < 0, -delta, 0.0)  # 负变化

        # 使用 pandas ewm (指数加权移动) 来计算平均增益和损失，比循环快得多
        # 注意：原代码是简单移动平均(SMA)，但 RSI 通常用指数移动平均(EMA)。这里按原逻辑用 SMA。
        # 如果想用标准 RSI，应使用 ewm(alpha=1/14)
        avg_gain_series = pd.Series(gain).rolling(window=14, min_periods=1).mean()
        avg_loss_series = pd.Series(loss).rolling(window=14, min_periods=1).mean()

        # 转换回 numpy 数组
        avg_gain = avg_gain_series.values
        avg_loss = avg_loss_series.values

        # 计算相对强度(RS)和RSI
        rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=(avg_loss != 0))
        rsi = 100 - 100 / (1 + rs)
        # 处理NaN，用50填充（中性值）
        rsi = np.nan_to_num(rsi, nan=50.0)

        # ------------------ 计算MACD指标 ------------------
        # 使用 pandas ewm 计算 EMA，比循环快得多
        close_series = pd.Series(close)
        ema12_series = close_series.ewm(span=12, adjust=False).mean()
        ema26_series = close_series.ewm(span=26, adjust=False).mean()

        # 转换回 numpy 数组
        ema12 = ema12_series.values
        ema26 = ema26_series.values

        # 计算MACD线
        macd = ema12 - ema26
        # 处理NaN
        macd = np.nan_to_num(macd, nan=0.0)

        # ------------------ 计算平均真实波幅(ATR) ------------------
        # 计算真实波幅(TR)
        tr0 = high - low
        tr1 = np.abs(high - np.roll(close, 1))  # np.roll 将 close 向后移动一位
        tr2 = np.abs(low - np.roll(close, 1))
        tr = np.max([tr0, tr1, tr2], axis=0)  # 在三个值中取最大

        # 使用 pandas rolling 计算 ATR
        atr_series = pd.Series(tr).rolling(window=14, min_periods=1).mean()
        atr = atr_series.values
        # 处理NaN
        atr = np.nan_to_num(atr, nan=0.0)

        # ------------------ 计算买入量比例 ------------------
        # 避免除零，当 volume 为0时，使用0.5作为默认值
        buy_ratio = np.divide(buy_volume, volume, out=np.full_like(buy_volume, 0.5), where=(volume != 0))

        # ------------------ 计算VWAP偏离度 ------------------
        # 避免除零，当 vwap 为0时，分母用1.0
        denominator = np.where(vwap != 0, vwap, 1.0)
        vwap_deviation = (close - vwap) / denominator
        # 处理无穷大和NaN值
        vwap_deviation = np.where(np.isfinite(vwap_deviation), vwap_deviation, 0.0)

        # ------------------ 将计算结果写回 DataFrame ------------------
        # 将所有计算好的因子作为新列添加到原始 DataFrame
        df['rsi'] = rsi
        df['macd'] = macd
        df['atr'] = atr
        df['buy_ratio'] = buy_ratio
        df['vwap_deviation'] = vwap_deviation

        return df

    # 读取每个币种的k线，然后计算成交量加权平均价(VWAP)【一种基本策略所需的指标】
    def get_single_symbol_kline_data(self, symbol):
        try:
            # 读取指定加密货币的Parquet文件，获取其K线数据作为DataFrame
            df = pd.read_parquet(f"{self.train_data_path}/{symbol}.parquet")
            # 将DataFrame的索引设置为"timestamp"列
            df = df.set_index("timestamp")
            # 将数据转换为64位浮点类型
            df = df.astype(np.float64)
            # 计算成交量加权平均价(VWAP)，处理无限值并用前一个有效值填充
            df['vwap'] = (df['amount'] / df['volume']).replace([np.inf, -np.inf], np.nan).ffill()
            df = self.compute_factors_pandas(df)
            print(f"Loaded data for {symbol}, shape: {df.shape}, vwap NaNs: {df['vwap'].isna().sum()}")
        except Exception as e:
            # ❌ 修复：不要返回空的 DataFrame
            print(f"Failed to load data for {symbol}: {e}")
            # ✅ 返回一个结构正确但为空的 DataFrame
            # 假设正常数据有这些列
            columns = ['open_price', 'high_price', 'low_price', 'close_price',
                       'volume', 'amount', 'buy_volume', 'vwap', 'rsi', 'macd',
                       'atr', 'buy_ratio', 'vwap_deviation']
            df = pd.DataFrame(columns=columns)
            df.index.name = 'timestamp'
            df = df.astype(np.float64)
        return df

    def get_all_symbol_kline(self):
        """时间点索引 | 币种索引 | 开盘价 | 最高价 | 最低价 | 收盘价 | VWAP | 交易额
        ----------+----------+--------+--------+--------+--------+------+------
            0     |    0     | 50000  | 50500  | 49900  | 50200  | 5000 | 5e6
            0     |    1     |  3500  |  3550  |  3490  |  3520  | 3505 | 2.5e6
            0     |    2     |   0.45 |   0.46 |   0.44 |   0.45 | 0.45 | 1e5
            1     |    0     | 50200  | 50300  | 50100  | 50250  | 5025 | 4e6
            1     |    1     |  3510  |  3520  |  3500  |  3515  | 3512 | 2.6e6
            1     |    2     |   0.46 |   0.47 |   0.45 |   0.46 | 0.46 | 1.2e5
            ...   |   ...    |  ...   |  ...   |  ...   |  ...   | ...  | ..."""
        t0 = datetime.datetime.now()
        with mp.Pool(4) as pool:
            all_symbol_list = self.get_all_symbol_list()
            # 提交所有任务
            results = [pool.apply_async(self.get_single_symbol_kline_data, (symbol,)) for symbol in all_symbol_list]
            # ✅ 关键：只调用一次 get() 来获取所有结果
            dfs = [r.get() for r in results]  # 这是一个 DataFrame 列表

        # 可选：检查并打印警告
        for symbol, df in zip(all_symbol_list, dfs):
            if df.empty:
                print(f"Warning: Data for {symbol} is empty, it will be filled with NaN.")
        # # 收集所有异步结果的开盘价序列并按列连接成DataFrame，然后按时间升序排序
        # df_open_price = pd.concat([i.get()['open_price'] for i in df_list], axis=1).sort_index(ascending=True)
        # # 将时间索引（毫秒）转换为datetime类型数组
        # time_arr = pd.to_datetime(pd.Series(df_open_price.index), unit="ms").values
        # # 从DataFrame中获取开盘价的值并转换为float类型的NumPy数组
        # open_price_arr = df_open_price.values.astype(float)
        # # 从DataFrame中获取最高价的值并转换为float类型的NumPy数组
        # high_price_arr = pd.concat([i.get()['high_price'] for i in df_list], axis=1).sort_index(ascending=True).values
        # # 从DataFrame中获取最低价的值并转换为float类型的NumPy数组
        # low_price_arr = pd.concat([i.get()['low_price'] for i in df_list], axis=1).sort_index(ascending=True).values
        # # 从DataFrame中获取收盘价的值并转换为float类型的NumPy数组
        # close_price_arr = pd.concat([i.get()['close_price'] for i in df_list], axis=1).sort_index(ascending=True).values
        # # 收集所有货币的成交量加权平均价格序列并按列连接成数组
        # vwap_arr = pd.concat([i.get()['vwap'] for i in df_list], axis=1).sort_index(ascending=True).values
        # # 收集所有货币的交易量序列并按列连接成数组
        # amount_arr = pd.concat([i.get()['amount'] for i in df_list], axis=1).sort_index(ascending=True).values
        # # 买方交易量拼接
        # buy_volume_arr = pd.concat([i.get()['buy_volume'] for i in df_list], axis=1).sort_index(ascending=True).values
        # # 基础交易量拼接
        # volume_arr = pd.concat([i.get()['volume'] for i in df_list], axis=1).sort_index(ascending=True).values
        # # rsi指标拼接
        # rsi_arr = pd.concat([i.get()['rsi'] for i in df_list], axis=1).sort_index(ascending=True).values
        # # macd指标拼接
        # macd_arr = pd.concat([i.get()['macd'] for i in df_list], axis=1).sort_index(ascending=True).values
        # # atr指标拼接
        # atr_arr = pd.concat([i.get()['atr'] for i in df_list], axis=1).sort_index(ascending=True).values
        # # buy_ratio指标拼接
        # buy_ratio_arr = pd.concat([i.get()['buy_ratio'] for i in df_list], axis=1).sort_index(ascending=True).values
        # # vwap-deviation指标拼接
        # vwap_deviation_arr = pd.concat([i.get()['vwap_deviation'] for i in df_list], axis=1).sort_index(ascending=True).values
        df_open_price = pd.concat([df['open_price'] for df in dfs], axis=1).sort_index(ascending=True)
        time_arr = pd.to_datetime(df_open_price.index, unit="ms").values
        open_price_arr = df_open_price.values.astype(float)
        high_price_arr = pd.concat([df['high_price'] for df in dfs], axis=1).sort_index(ascending=True).values
        low_price_arr = pd.concat([df['low_price'] for df in dfs], axis=1).sort_index(ascending=True).values
        close_price_arr = pd.concat([df['close_price'] for df in dfs], axis=1).sort_index(ascending=True).values
        vwap_arr = pd.concat([df['vwap'] for df in dfs], axis=1).sort_index(ascending=True).values
        amount_arr = pd.concat([df['amount'] for df in dfs], axis=1).sort_index(ascending=True).values
        buy_volume_arr = pd.concat([df['buy_volume'] for df in dfs], axis=1).sort_index(ascending=True).values
        volume_arr = pd.concat([df['volume'] for df in dfs], axis=1).sort_index(ascending=True).values
        rsi_arr = pd.concat([df['rsi'] for df in dfs], axis=1).sort_index(ascending=True).values
        macd_arr = pd.concat([df['macd'] for df in dfs], axis=1).sort_index(ascending=True).values
        atr_arr = pd.concat([df['atr'] for i in dfs], axis=1).sort_index(ascending=True).values
        buy_ratio_arr = pd.concat([df['buy_ratio'] for df in dfs], axis=1).sort_index(ascending=True).values
        vwap_deviation_arr = pd.concat([df['vwap_deviation'] for df in dfs], axis=1).sort_index(ascending=True).values

        print(f"完成获取所有币种k线数据, 耗时 {datetime.datetime.now() - t0}")
        return (all_symbol_list, time_arr, open_price_arr, high_price_arr, low_price_arr,
                close_price_arr, vwap_arr, amount_arr, buy_volume_arr, volume_arr, rsi_arr,
                macd_arr, atr_arr, buy_ratio_arr, vwap_deviation_arr)

    def weighted_spearmanr(self, y_true, y_pred):
        """
        根据附录公式计算 加权Spearman相关系数：
        1) 将y_true和y_pred按降序排列（rank=1表示最大值）
        2) 将排名索引归一化到[-1, 1]范围，然后平方得到权重w_i
        3) 使用加权Pearson公式计算相关系数

        反映了模型预测排名与实际排名之间的一致性程度，特别强调了排名靠前（高收益率）币种的准确性。
        值越靠近1，说明模型在预测方面与实际收益的贴合程度（并且主要关注排名靠前的虚拟币）
        """
        # 样本数量
        n = len(y_true)
        # 对真实值进行降序排列（处理相同值时使用平均法）
        r_true = pd.Series(y_true).rank(ascending=False, method='average')
        # 对预测值进行降序排列（处理相同值时使用平均法）
        r_pred = pd.Series(y_pred).rank(ascending=False, method='average')

        # 将索引i = rank - 1归一化到[-1, 1]范围
        x = 2 * (r_true - 1) / (n - 1) - 1
        # 权重w_i（每个样本的权重因子）
        w = x ** 2

        # 加权平均值计算
        w_sum = w.sum()
        mu_true = (w * r_true).sum() / w_sum
        mu_pred = (w * r_pred).sum() / w_sum

        # 计算加权协方差
        cov = (w * (r_true - mu_true) * (r_pred - mu_pred)).sum()
        # 计算真实值排名的加权方差
        var_true = (w * (r_true - mu_true) ** 2).sum()
        # 计算预测值排名的加权方差
        var_pred = (w * (r_pred - mu_pred) ** 2).sum()

        # 返回加权Spearman相关系数
        return cov / np.sqrt(var_true * var_pred)

    class LSTMModel(nn.Module):
        def __init__(self, input_size, hidden_size=64, num_layers=2, output_size=1, dropout=0.2):
            super(LSTMModel, self).__init__()
            self.hidden_size = hidden_size
            self.num_layers = num_layers

            self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                                batch_first=True, dropout=dropout)
            self.dropout = nn.Dropout(dropout)
            self.fc = nn.Linear(hidden_size, output_size)

        def forward(self, x):
            # 初始化隐藏状态和细胞状态
            h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
            c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

            # LSTM 前向传播
            out, _ = self.lstm(x, (h0, c0))

            # 取最后一个时间步的输出
            out = self.dropout(out[:, -1, :])
            out = self.fc(out)
            return out

    # def train(self, df_target, *df_factors):
    #     """
    #     优化内存的训练方法。
    #     参数:
    #     df_target (pd.DataFrame): 目标值 DataFrame (时间 x 符号)。
    #     *df_factors (pd.DataFrame): 可变数量的因子 DataFrame (时间 x 符号)。
    #     """
    #     print("--- 开始 train 方法 (优化版) ---")
    #     train_start_time = datetime.datetime.now()

    #     # --- 验证输入 ---
    #     if len(df_factors) == 0:
    #         print("错误：未提供任何因子数据。")
    #         return

    #     num_factors = len(df_factors)
    #     print(f"  接收到 {num_factors} 个因子。")

    #     # 假设所有 DataFrame 的索引和列都一致
    #     all_symbol_list = df_target.columns.tolist()
    #     time_index = df_target.index  # 假设所有 df 都有相同的时间索引
    #     print(f"  检测到 {len(all_symbol_list)} 个符号。")

    #     # ------------------ 数据分割 ------------------
    #     start_datetime = self.start_datetime
    #     print(f"  数据分割点 (start_datetime): {start_datetime}")

    #     # 分割时间索引
    #     train_time_mask = time_index < start_datetime
    #     pred_time_mask = time_index >= start_datetime

    #     train_times = time_index[train_time_mask]
    #     pred_times = time_index[pred_time_mask]

    #     if len(train_times) == 0:
    #         print("  警告：训练时间范围为空！检查 start_datetime 设置。")
    #         return

    #     print(f"  训练时间点数: {len(train_times)}")
    #     print(f"  预测时间点数: {len(pred_times)}")

    #     # --- 准备因子和目标数据 (保持 DataFrame 形式，按列组织) ---
    #     # 目标值
    #     target_data = df_target.values  # 转换为 numpy 数组以提高效率
    #     # 因子数据 (列表中的每个元素是一个因子的二维数组)
    #     factors_data_list = [f.values for f in df_factors]  # List of 2D arrays (time x symbols)

    #     # --- 构建训练序列 (流式处理) ---
    #     seq_length = 24 * 4 * 3
    #     min_sequence_length = seq_length + 10
    #     max_total_sequences = 500000
    #     max_sequences_per_symbol = 1000

    #     print("  开始为训练集构建时序序列 (流式处理)...")
    #     train_seq_start_time = datetime.datetime.now()

    #     # 预先分配列表，避免动态增长
    #     X_train_seq_list = []
    #     y_train_seq_list = []
    #     total_train_sequences = 0

    #     # --- 按 Symbol 处理训练数据 ---
    #     for symbol_idx, symbol in enumerate(all_symbol_list):
    #         if total_train_sequences >= max_total_sequences:
    #             print(f"    达到最大训练序列数限制: {max_total_sequences}")
    #             break

    #         # 1. 提取单个 Symbol 的数据 (时间序列)
    #         #    注意：.values 返回的是视图，不复制数据，效率高
    #         symbol_target = target_data[:, symbol_idx]  # 1D array (time,)
    #         symbol_factors_list = [f[:, symbol_idx] for f in factors_data_list]  # List of 1D arrays (time,)

    #         # 2. 创建 DataFrame 以便于处理 (只包含该 symbol 的数据)
    #         #    这个 DataFrame 是临时的，只用于当前 symbol
    #         symbol_train_data_dict = {f'factor_{i}': symbol_factors_list[i][train_time_mask] for i in
    #                                   range(num_factors)}
    #         symbol_train_data_dict['target'] = symbol_target[train_time_mask]

    #         symbol_train_df = pd.DataFrame(symbol_train_data_dict, index=train_times)
    #         # print(f"    Symbol {symbol}: 训练数据形状 {symbol_train_df.shape}") # 调试用

    #         # 3. 检查长度
    #         n_train_points = len(symbol_train_df)
    #         if n_train_points <= min_sequence_length:
    #             # print(f"    Symbol {symbol}: 数据不足 ({n_train_points} <= {min_sequence_length}), 跳过。") # 调试用
    #             continue

    #         # 4. 提取因子和目标值的 numpy 数组 (用于滑动窗口)
    #         factor_values = symbol_train_df[
    #             [f'factor_{i}' for i in range(num_factors)]].values  # 2D array (time, factors)
    #         target_values = symbol_train_df['target'].values  # 1D array (time,)

    #         # 5. 构建滑动窗口序列
    #         n_sequences = min(n_train_points - seq_length, max_sequences_per_symbol)
    #         for i in range(n_sequences):
    #             if total_train_sequences >= max_total_sequences:
    #                 break
    #             # X: [seq_length, num_factors]
    #             X_train_seq_list.append(factor_values[i:(i + seq_length)])
    #             # y: scalar (目标是序列最后一个点的值)
    #             y_train_seq_list.append(target_values[i + seq_length - 1])
    #             total_train_sequences += 1

    #         # 6. (可选) 定期清理临时 DataFrame
    #         # del symbol_train_df, symbol_target, symbol_factors_list, factor_values, target_values
    #         # gc.collect() # 通常不需要，Python 会自动处理

    #     train_seq_end_time = datetime.datetime.now()
    #     print(f"  训练序列构建完成，共 {total_train_sequences} 个序列，耗时 {train_seq_end_time - train_seq_start_time}")

    #     if total_train_sequences == 0:
    #         print("  没有足够的训练数据构建序列")
    #         return

    #     # --- 转换为最终的 NumPy 数组 ---
    #     print("  转换训练序列为 NumPy 数组...")
    #     X_train_sequences = np.array(X_train_seq_list)  # Shape: (num_seq, seq_len, num_factors)
    #     y_train_sequences = np.array(y_train_seq_list)  # Shape: (num_seq,)
    #     print(f"  训练集序列形状: X_train={X_train_sequences.shape}, y_train={y_train_sequences.shape}")

    #     # --- 清理中间列表 ---
    #     del X_train_seq_list, y_train_seq_list
    #     gc.collect()

    #     # --- 数据标准化 (优化版，加强清洗和内存管理) ---
    #     print("  开始训练集数据标准化...")
    #     scaler_X = StandardScaler()
    #     scaler_y = StandardScaler()

    #     # --- 加强数据清洗 ---
    #     print("    清洗训练数据中的 NaN 和 Inf...")
    #     # 检查 X_train_sequences 是否包含 NaN 或 Inf
    #     print(f"    X_train_sequences NaN count before cleaning: {np.isnan(X_train_sequences).sum()}")
    #     print(f"    X_train_sequences Inf count before cleaning: {np.isinf(X_train_sequences).sum()}")
    #     print(f"    y_train_sequences NaN count before cleaning: {np.isnan(y_train_sequences).sum()}")
    #     print(f"    y_train_sequences Inf count before cleaning: {np.isinf(y_train_sequences).sum()}")

    #     # 将 Inf 和 -Inf 替换为 NaN，然后将 NaN 替换为 0
    #     X_train_sequences = np.nan_to_num(X_train_sequences, nan=0.0, posinf=0.0, neginf=0.0)
    #     y_train_sequences = np.nan_to_num(y_train_sequences, nan=0.0, posinf=0.0, neginf=0.0)

    #     print(f"    X_train_sequences NaN count after cleaning: {np.isnan(X_train_sequences).sum()}")
    #     print(f"    X_train_sequences Inf count after cleaning: {np.isinf(X_train_sequences).sum()}")
    #     print(f"    y_train_sequences NaN count after cleaning: {np.isnan(y_train_sequences).sum()}")
    #     print(f"    y_train_sequences Inf count after cleaning: {np.isinf(y_train_sequences).sum()}")
    #     # --- 数据清洗结束 ---

    #     # --- 优化标准化过程 ---
    #     batch_size = 500  # <<<<<<<<< 减小 batch_size
    #     print(f"    使用 batch_size: {batch_size}")
    #     num_samples = X_train_sequences.shape[0]
    #     num_features = X_train_sequences.shape[2]

    #     # 预分配标准化后的数组，使用 float32
    #     X_train_scaled = np.empty((num_samples, X_train_sequences.shape[1], num_features), dtype=np.float32)
    #     # y 是 1D 的
    #     y_train_scaled = np.empty((num_samples,), dtype=np.float32)

    #     print("    分批进行标准化...")
    #     for i in range(0, num_samples, batch_size):
    #         end_idx = min(i + batch_size, num_samples)
    #         current_batch_size = end_idx - i
    #         if current_batch_size == 0:  # 防御性检查
    #             continue

    #         # 1. 提取批次 (视图，不复制)
    #         batch_X = X_train_sequences[i:end_idx]
    #         batch_y = y_train_sequences[i:end_idx]

    #         # 2. 重塑批次用于 StandardScaler: (batch_size * seq_len, num_features)
    #         original_shape_X = batch_X.shape
    #         batch_X_reshaped = batch_X.reshape(-1, num_features).astype(np.float32, copy=False)  # copy=False 尝试避免复制

    #         # 3. 标准化 X
    #         if i == 0:
    #             # 第一个批次：拟合并变换
    #             batch_X_scaled_reshaped = scaler_X.fit_transform(batch_X_reshaped)
    #         else:
    #             # 后续批次：仅变换
    #             batch_X_scaled_reshaped = scaler_X.transform(batch_X_reshaped)

    #         # 4. 恢复 X 的形状并存入预分配数组
    #         batch_X_scaled = batch_X_scaled_reshaped.reshape(original_shape_X)
    #         X_train_scaled[i:end_idx] = batch_X_scaled.astype(np.float32, copy=False)

    #         # 5. 标准化 y (1D)
    #         batch_y_reshaped = batch_y.reshape(-1, 1).astype(np.float32, copy=False)
    #         if i == 0:
    #             # 第一个批次：拟合并变换
    #             batch_y_scaled_reshaped = scaler_y.fit_transform(batch_y_reshaped)
    #         else:
    #             # 后续批次：仅变换
    #             batch_y_scaled_reshaped = scaler_y.transform(batch_y_reshaped)

    #         # 6. 恢复 y 的形状并存入预分配数组
    #         batch_y_scaled = batch_y_scaled_reshaped.flatten()  # 变回 1D
    #         y_train_scaled[i:end_idx] = batch_y_scaled.astype(np.float32, copy=False)

    #         # 7. (可选) 显式删除中间变量并清理内存
    #         # del batch_X, batch_y, batch_X_reshaped, batch_X_scaled_reshaped, batch_X_scaled, batch_y_reshaped, batch_y_scaled_reshaped, batch_y_scaled
    #         # gc.collect()

    #         # 8. 打印进度
    #         if (i // batch_size) % 50 == 0 or end_idx == num_samples:  # 每50个批次或最后打印一次
    #             print(f"      已处理 {end_idx}/{num_samples} 个样本 ({100 * end_idx / num_samples:.1f}%)")

    #     print("  训练集数据标准化完成")
    #     # --- 优化标准化过程结束 ---

    #     # --- 清理原始序列并强制垃圾回收 ---
    #     print("  清理原始训练序列数据以释放内存...")
    #     del X_train_sequences, y_train_sequences  # 删除原始大数组
    #     gc.collect()  # 强制进行垃圾回收
    #     print("  原始训练序列数据清理完成")
    #     # --- 清理结束 ---

    #     # --- 转换为 PyTorch 张量 ---
    #     print("  转换训练集为 PyTorch 张量...")
    #     device = torch.device(self.device)  # 使用类实例中预先确定的设备
    #     print(f"  Training tensors will be placed on: {device}")

    #     # 2. 确保数据已经是 float32 (在标准化后应该已经是了，但再次确认)
    #     # 3. 使用 torch.from_numpy 转换，它更高效且与 numpy 数组共享内存 (直到 .to(device))
    #     # 4. 使用 .contiguous() 确保内存连续，这对某些 GPU 操作很重要
    #     X_train_tensor = torch.from_numpy(X_train_scaled).contiguous().to(device,
    #                                                                       non_blocking=True)  # non_blocking 对于异步传输可能有帮助
    #     y_train_tensor = torch.from_numpy(y_train_scaled).contiguous().to(device, non_blocking=True)

    #     # 5. 立即删除 numpy 数组副本以尝试释放内存
    #     del X_train_scaled, y_train_scaled
    #     gc.collect()
    #     print("  PyTorch 张量转换完成")
    #     # --- 转换结束 ---

    #     # ... (后续代码保持不变，创建 dataset, dataloader, model, training loop 等) ...

    #     train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    #     # 注意：DataLoader 会复制数据到设备，如果内存紧张，可以考虑在循环中手动移动 batch
    #     train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    #     print("  初始化 LSTM 模型...")
    #     # --- 关键修改：确保 input_size 与因子数量匹配 ---
    #     model = LSTMModel(input_size=num_factors, hidden_size=64, num_layers=2, output_size=1, dropout=0.2).to(device)
    #     print(f"  模型输入维度: {num_factors}")

    #     criterion = nn.MSELoss()
    #     optimizer = optim.Adam(model.parameters(), lr=0.001)
    #     scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)

    #     print("  开始训练模型...")
    #     model.train()
    #     num_epochs = 100
    #     for epoch in range(num_epochs):
    #         epoch_start_time = datetime.datetime.now()
    #         total_loss = 0
    #         for batch_X, batch_y in train_dataloader:
    #             # batch_X, batch_y = batch_X.to(device), batch_y.to(device) # DataLoader已处理
    #             optimizer.zero_grad()
    #             outputs = model(batch_X)
    #             loss = criterion(outputs.squeeze(), batch_y)
    #             loss.backward()
    #             torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    #             optimizer.step()
    #             total_loss += loss.item()
    #         scheduler.step()
    #         epoch_end_time = datetime.datetime.now()
    #         if (epoch + 1) % 10 == 0:
    #             print(
    #                 f'    Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss / len(train_dataloader):.6f}, '
    #                 f'LR: {scheduler.get_last_lr()[0]:.6f}, Time: {epoch_end_time - epoch_start_time}')

    #     print("  模型训练完成")

    #     # --- 构建预测序列 (流式处理) ---
    #     print("  开始为预测集构建时序序列 (流式处理)...")
    #     pred_seq_start_time = datetime.datetime.now()

    #     X_pred_seq_list = []
    #     pred_index_info = []  # 存储 (datetime, symbol) 用于结果映射

    #     # --- 按 Symbol 处理预测数据 ---
    #     for symbol_idx, symbol in enumerate(all_symbol_list):
    #         # 1. 提取单个 Symbol 的数据
    #         symbol_target = target_data[:, symbol_idx]  # 1D array (time,)
    #         symbol_factors_list = [f[:, symbol_idx] for f in factors_data_list]  # List of 1D arrays (time,)

    #         # 2. 创建 DataFrame 以便于处理 (只包含该 symbol 的数据)
    #         symbol_pred_data_dict = {f'factor_{i}': symbol_factors_list[i][pred_time_mask] for i in range(num_factors)}
    #         symbol_pred_data_dict['target'] = symbol_target[pred_time_mask]  # 真实目标值也包含，用于 check.csv

    #         symbol_pred_df = pd.DataFrame(symbol_pred_data_dict, index=pred_times)
    #         # print(f"    Symbol {symbol}: 预测数据形状 {symbol_pred_df.shape}") # 调试用

    #         # 3. 检查长度
    #         n_pred_points = len(symbol_pred_df)
    #         if n_pred_points < seq_length:
    #             # print(f"    Symbol {symbol}: 预测数据不足 ({n_pred_points} < {seq_length}), 跳过。") # 调试用
    #             continue

    #         # 4. 提取因子的 numpy 数组
    #         factor_values = symbol_pred_df[
    #             [f'factor_{i}' for i in range(num_factors)]].values  # 2D array (time, factors)

    #         # 5. 构建一个序列：使用最后 seq_length 个时间点的数据进行预测
    #         # X: [1, seq_length, num_factors]
    #         X_pred_seq_list.append(factor_values[-seq_length:])
    #         # 记录这个序列对应的 datetime 和 symbol (使用最后一个时间点)
    #         pred_datetime = symbol_pred_df.index[-1]
    #         pred_index_info.append((pred_datetime, symbol))

    #     pred_seq_end_time = datetime.datetime.now()
    #     print(f"  预测序列构建完成，共 {len(X_pred_seq_list)} 个序列，耗时 {pred_seq_end_time - pred_seq_start_time}")

    #     if len(X_pred_seq_list) == 0:
    #         print("  没有足够的预测数据构建序列")
    #         self._create_empty_submission()
    #         return

    #     # --- 转换为最终的 NumPy 数组 ---
    #     print("  转换预测序列为 NumPy 数组...")
    #     X_pred_sequences = np.array(X_pred_seq_list)  # Shape: (num_pred_seq, seq_len, num_factors)
    #     print(f"  预测集序列形状: X_pred={X_pred_sequences.shape}")

    #     # --- 清理中间列表 ---
    #     del X_pred_seq_list
    #     gc.collect()

    #     # --- 预测集数据标准化 ---
    #     print("  开始预测集数据标准化...")
    #     # 重塑 X 用于标准化: (num_pred_seq * seq_len, num_factors)
    #     original_shape_X_pred = X_pred_sequences.shape
    #     X_pred_reshaped = X_pred_sequences.reshape(-1, original_shape_X_pred[-1])
    #     X_pred_scaled = scaler_X.transform(X_pred_reshaped)  # 使用训练集的 scaler
    #     # 恢复形状
    #     X_pred_scaled = X_pred_scaled.reshape(original_shape_X_pred)
    #     print("  预测集数据标准化完成")

    #     # --- 转换为 PyTorch 张量并预测 ---
    #     print("  转换预测集为 PyTorch 张量...")
    #     X_pred_tensor = torch.FloatTensor(X_pred_scaled).to(device)
    #     del X_pred_scaled, X_pred_reshaped

    #     print("  开始对预测集进行预测...")
    #     model.eval()
    #     y_pred_list = []
    #     with torch.no_grad():
    #         for i in range(0, len(X_pred_tensor), 1000):  # 分批预测
    #             batch_X = X_pred_tensor[i:i + 1000]  # .to(device) 已在创建时完成
    #             batch_pred = model(batch_X).cpu().numpy().flatten()
    #             y_pred_list.append(batch_pred)

    #     if y_pred_list:
    #         y_pred_scaled_final = np.concatenate(y_pred_list)
    #         # 使用训练集的 y_scaler 进行逆变换
    #         y_pred_final = scaler_y.inverse_transform(y_pred_scaled_final.reshape(-1, 1)).flatten()
    #         print("  预测集预测完成")
    #     else:
    #         print("  预测列表为空")
    #         y_pred_final = np.array([])

    #     # --- 清理 GPU 内存 ---
    #     del X_train_tensor, y_train_tensor, train_dataset, train_dataloader
    #     del X_pred_tensor
    #     if torch.cuda.is_available():
    #         torch.cuda.empty_cache()
    #     gc.collect()

    #     # --- 生成最终提交文件 ---
    #     self._generate_submission_files(y_pred_final, pred_index_info, time_index, all_symbol_list, target_data)

    #     train_end_time = datetime.datetime.now()
    #     print(f"--- train 方法结束，总耗时: {train_end_time - train_start_time} ---")

    # def _create_empty_submission(self):
    #     """辅助函数：创建空的提交文件"""
    #     print("  创建空的提交文件...")
    #     df_submission_id = pd.read_csv("submission_id.csv")
    #     df_submit_competition = df_submission_id.copy()
    #     df_submit_competition['predict_return'] = 0
    #     df_submit_competition.to_csv("submit.csv", index=False)
    #     empty_check = pd.DataFrame(columns=['id', 'true_return'])
    #     empty_check.to_csv("check.csv", index=False)

    # def _generate_submission_files(self, y_pred_final, pred_index_info, time_index, all_symbol_list, target_data):
    #     """辅助函数：生成最终的 submit.csv 和 check.csv"""
    #     if len(y_pred_final) > 0 and len(pred_index_info) == len(y_pred_final):
    #         print("  开始生成最终提交文件...")

    #         # --- 生成 submit.csv ---
    #         final_pred_data_list = []
    #         for i, (dt, symbol) in enumerate(pred_index_info):
    #             final_pred_data_list.append({
    #                 'datetime': dt,
    #                 'symbol': symbol,
    #                 'predict_return': y_pred_final[i]
    #             })
    #         df_final_submit = pd.DataFrame(final_pred_data_list)
    #         df_final_submit["id"] = df_final_submit["datetime"].astype(str) + "_" + df_final_submit["symbol"]
    #         df_final_submit = df_final_submit[['id', 'predict_return']]
    #         print("  最终预测结果样本:")
    #         print(df_final_submit.head())

    #         # 读取提交ID列表并匹配
    #         df_submission_id = pd.read_csv("submission_id.csv")
    #         id_list = df_submission_id["id"].tolist()
    #         df_submit_competition = df_final_submit[df_final_submit['id'].isin(id_list)]
    #         # 添加缺失的ID（用0填充）
    #         missing_elements = list(set(id_list) - set(df_submit_competition['id']))
    #         if missing_elements:
    #             new_rows = pd.DataFrame({'id': missing_elements, 'predict_return': [0] * len(missing_elements)})
    #             df_submit_competition = pd.concat([df_submit_competition, new_rows], ignore_index=True)
    #         print(f"  最终提交数据形状: {df_submit_competition.shape}")
    #         df_submit_competition.to_csv("submit.csv", index=False)

    #         # --- 生成 check.csv ---
    #         check_data_list = []
    #         # 创建一个 target_data 的 DataFrame 便于查找
    #         df_full_target = pd.DataFrame(target_data, index=time_index, columns=all_symbol_list)
    #         for i, (dt, symbol) in enumerate(pred_index_info):
    #             # 从完整的 target DataFrame 中找到对应的真实目标值
    #             try:
    #                 # 使用 .loc 进行精确查找
    #                 true_return = df_full_target.loc[dt, symbol]
    #                 if pd.isna(true_return):
    #                     true_return = 0  # 如果是 NaN，用 0 填充
    #                 check_data_list.append({
    #                     'id': f"{dt}_{symbol}",
    #                     'true_return': true_return
    #                 })
    #             except KeyError:
    #                 # print(f"  警告：在 target 数据中未找到 ({dt}, {symbol})，用 0 填充。")
    #                 check_data_list.append({
    #                     'id': f"{dt}_{symbol}",
    #                     'true_return': 0
    #                 })
    #         df_check = pd.DataFrame(check_data_list)
    #         df_check.to_csv("check.csv", index=False)
    #         print("  检查文件已生成 (包含预测时间点的真实值)")

    #         # --- 计算 Spearman 系数 ---
    #         # ... (之前的检查 len(check_data_list) > 0 等保持不变) ...
    #         if len(check_data_list) > 0:
    #             check_df_for_corr = pd.DataFrame(check_data_list)
    #             # 确保 df_final_submit 也有 'predict_return' 列
    #             if not check_df_for_corr.empty and 'true_return' in check_df_for_corr.columns and 'predict_return' in df_final_submit.columns:
    #                 # 1. 合并预测值和真实值
    #                 merged_for_corr = pd.merge(df_final_submit[['id', 'predict_return']],
    #                                            check_df_for_corr[['id', 'true_return']], on='id', how='inner')

    #                 if len(merged_for_corr) > 0:
    #                     try:
    #                         print(f"    Debug: 准备计算Spearman相关系数...")
    #                         print(f"    Debug: merged_for_corr shape (合并后): {merged_for_corr.shape}")

    #                         # 2. 处理 NaN: 对于 Spearman 计算，必须成对地移除 NaN
    #                         #    dropna(subset=[...]) 会移除指定列中任一列为 NaN 的行
    #                         merged_for_corr_clean = merged_for_corr.dropna(subset=['true_return', 'predict_return'])

    #                         print(f"    Debug: merged_for_corr shape (清理NaN后): {merged_for_corr_clean.shape}")
    #                         print(
    #                             f"    Debug: true_return describe (清理后):\n{merged_for_corr_clean['true_return'].describe()}")
    #                         print(
    #                             f"    Debug: predict_return describe (清理后):\n{merged_for_corr_clean['predict_return'].describe()}")

    #                         # 3. 检查清理后数据是否足够
    #                         if len(merged_for_corr_clean) > 0:
    #                             y_true_clean = merged_for_corr_clean['true_return']
    #                             y_pred_clean = merged_for_corr_clean['predict_return']

    #                             # 4. 检查值是否恒定 (这是导致 NaN 的常见原因)
    #                             if y_true_clean.nunique() <= 1:
    #                                 print("    Warning: true_return 中所有值都相同或只有一个唯一值，无法计算相关系数。")
    #                                 print(f"             Unique values in true_return: {y_true_clean.unique()}")
    #                                 rho_overall = np.nan  # 或 0
    #                             elif y_pred_clean.nunique() <= 1:
    #                                 print(
    #                                     "    Warning: predict_return 中所有值都相同或只有一个唯一值，无法计算相关系数。")
    #                                 print(f"             Unique values in predict_return: {y_pred_clean.unique()}")
    #                                 rho_overall = np.nan  # 或 0
    #                             else:
    #                                 # 5. 执行计算
    #                                 rho_overall = self.weighted_spearmanr(y_true_clean, y_pred_clean)
    #                                 print(f"  预测集加权Spearman相关系数: {rho_overall:.4f}")
    #                         else:
    #                             print("    Error: 清理 NaN 后没有有效数据对用于计算 Spearman 系数。")
    #                             rho_overall = np.nan

    #                     except Exception as e:
    #                         print(f"  计算Spearman相关系数时出错: {e}")
    #                         import traceback
    #                         traceback.print_exc()
    #                 else:
    #                     print("  无法计算Spearman相关系数：合并后数据为空 (inner join 结果为空)")
    #             else:
    #                 print(
    #                     "  无法计算Spearman相关系数：缺少必要列 (check_df 需要 'true_return', submit_df 需要 'predict_return')")
    #         else:
    #             print("  无法计算Spearman相关系数：检查数据为空 (check_data_list 为空)")
    #         # --- 计算 Spearman 系数结束 ---

    def run(self):
        print("--- 开始 run 方法 ---")
        run_start_time = datetime.datetime.now()
        # 调用get_all_symbol_kline函数获取所有货币的K线数据和事件数据
        print("  开始获取所有币种K线数据...")
        (all_symbol_list, time_arr, open_price_arr, high_price_arr,
         low_price_arr, close_price_arr, vwap_arr, amount_arr, buy_volume_arr, volume_arr,
         rsi_arr, macd_arr, atr_arr, buy_ratio_arr, vwap_deviation_arr) = self.get_all_symbol_kline()
        print("  所有币种K线数据获取完成")
        # 将vwap数组转换为DataFrame，货币代码作为列，时间作为索引（下一行设置索引）
        print("  开始转换为DataFrame...")
        df_vwap = pd.DataFrame(vwap_arr, columns=all_symbol_list, index=time_arr)
        # 将amount数组转换为DataFrame，货币代码作为列，时间作为索引
        df_amount = pd.DataFrame(amount_arr, columns=all_symbol_list, index=time_arr)
        df_buy_volume = pd.DataFrame(buy_volume_arr, columns=all_symbol_list, index=time_arr)
        df_volume = pd.DataFrame(volume_arr, columns=all_symbol_list, index=time_arr)
        df_open_price = pd.DataFrame(open_price_arr, columns=all_symbol_list, index=time_arr)
        df_high_price = pd.DataFrame(high_price_arr, columns=all_symbol_list, index=time_arr)
        df_low_price = pd.DataFrame(low_price_arr, columns=all_symbol_list, index=time_arr)
        df_close_price = pd.DataFrame(close_price_arr, columns=all_symbol_list, index=time_arr)
        df_rsi = pd.DataFrame(rsi_arr, columns=all_symbol_list, index=time_arr)
        df_macd = pd.DataFrame(macd_arr, columns=all_symbol_list, index=time_arr)
        df_atr = pd.DataFrame(atr_arr, columns=all_symbol_list, index=time_arr)
        df_buy_ratio = pd.DataFrame(buy_ratio_arr, columns=all_symbol_list, index=time_arr)
        df_vwap_deviation = pd.DataFrame(vwap_deviation_arr, columns=all_symbol_list, index=time_arr)
        print("  DataFrame转换完成")
        windows_1d = 4 * 24 * 1  # 1天对应的窗口大小（15分钟K线）
        windows_7d = 4 * 24 * 7  # 7天对应的窗口大小（15分钟K线）
        windows_4h = 4 * 4  # 4小时
        windows_14d = 4 * 24 * 14
        windows_20d = 4 * 24 * 20  # 新增20天窗口
        print("  开始计算特征...")

        # --- 计时开始：基本特征 ---
        basic_feature_start = datetime.datetime.now()
        # 基本特征
        # 单次波动
        price_range = df_high_price - df_close_price
        # 价格变化率
        change_rate = (df_close_price - df_open_price) / df_close_price
        # 每笔交易的平均金额
        averagetradingprice = df_buy_volume / df_amount
        # 上影线长度
        uppershadow = df_high_price - np.maximum(df_open_price, df_close_price)
        # 下影线长度
        lowershadow = np.minimum(df_open_price, df_close_price) - df_low_price
        # 买方交易量占比
        buyvolumnratio = df_buy_volume / df_volume
        # 买方交易额占比
        buyamountratio = df_buy_volume / df_amount
        # 卖方交易量
        sellvolume = df_volume - df_buy_volume
        # 买卖交易量比
        BSratio = df_buy_volume / sellvolume
        basic_feature_end = datetime.datetime.now()
        print(f"  基本特征计算耗时: {basic_feature_end - basic_feature_start}")
        # --- 计时结束：基本特征 ---

        # --- 计时开始：动量与交易量因子 ---
        momentum_vol_start = datetime.datetime.now()
        # 计算动量因子
        df_4h_momentum = df_vwap.div(df_vwap.shift(windows_4h)).sub(1).replace([np.inf, -np.inf], np.nan).fillna(0)
        df_7d_momentum = df_vwap.div(df_vwap.shift(windows_7d)).sub(1).replace([np.inf, -np.inf], np.nan).fillna(0)
        # 计算交易量因子
        df_amount_sum = df_amount.rolling(windows_7d).sum().replace([np.inf, -np.inf], np.nan).fillna(0)  # 7天总交易额
        # 计算交易额动量
        df_vol_momentum = df_amount.div(df_amount.shift(windows_1d)).sub(1).replace([np.inf, -np.inf], np.nan).fillna(0)
        # 计算市场压力因子
        df_buy_pressure = (df_buy_volume - (df_volume - df_buy_volume)).replace([np.inf, -np.inf], np.nan).fillna(
            0)  # 买入压力
        momentum_vol_end = datetime.datetime.now()
        print(f"  动量与交易量因子计算耗时: {momentum_vol_end - momentum_vol_start}")
        # --- 计时结束：动量与交易量因子 ---

        # --- 计时开始：收益率与波动率因子 ---
        return_volatility_start = datetime.datetime.now()
        # 使用滚动计算计算过去24小时的收益率
        df_24hour_rtn = df_vwap.div(df_vwap.shift(windows_1d)).sub(1)
        # 使用滚动计算计算过去15分钟的收益率
        df_15min_rtn = df_vwap.div(df_vwap.shift(1)).sub(1)
        # 计算第一个因子：7天波动率因子，反映市场风险和不确定性
        df_7d_volatility = df_15min_rtn.rolling(windows_7d).std(ddof=1)
        return_volatility_end = datetime.datetime.now()
        print(f"  收益率与波动率因子计算耗时: {return_volatility_end - return_volatility_start}")
        # --- 计时结束：收益率与波动率因子 ---

        # --- 计时开始：布林带因子 ---
        bb_start = datetime.datetime.now()
        # 布林带因子（使用收盘价计算）
        df_bb_ma = df_close_price.rolling(windows_20d).mean()
        df_bb_std = df_close_price.rolling(windows_20d).std()
        df_bb_upper = df_bb_ma + 2 * df_bb_std
        df_bb_lower = df_bb_ma - 2 * df_bb_std
        df_bb_width = df_bb_upper - df_bb_lower
        df_bb_position = (df_close_price - df_bb_lower).div(df_bb_width + 1e-6)  # 使用 div
        bb_end = datetime.datetime.now()
        print(f"  布林带因子计算耗时: {bb_end - bb_start}")
        # --- 计时结束：布林带因子 ---

        # --- 计时开始：OBV因子 ---
        obv_start = datetime.datetime.now()
        # OBV能量潮因子
        df_obv = (df_volume * np.sign(df_close_price.diff())).cumsum()
        df_obv_ma5 = df_obv.rolling(5).mean()
        df_obv_ma20 = df_obv.rolling(windows_20d).mean()
        df_obv_std20 = df_obv.rolling(windows_20d).std()
        df_obv_ratio = df_obv.div(df_obv_ma5).replace([np.inf, -np.inf], np.nan).fillna(0)
        df_obv_zscore = (df_obv - df_obv_ma20).div(df_obv_std20 + 1e-6)
        obv_end = datetime.datetime.now()
        print(f"  OBV因子计算耗时: {obv_end - obv_start}")
        # --- 计时结束：OBV因子 ---

        # --- 计时开始：价量相关性因子 ---
        corr_start = datetime.datetime.now()
        # 价量相关性因子
        df_corr = df_close_price.rolling(windows_20d).corr(df_volume)
        df_corr_ma = df_corr.rolling(windows_20d).mean()
        df_corr_diff = df_corr - df_corr_ma
        df_corr_extreme = pd.DataFrame(
            np.select(
                [df_corr > 0.7, df_corr < -0.5],
                [1, -1],
                default=0
            ),
            columns=all_symbol_list,
            index=time_arr
        )
        corr_end = datetime.datetime.now()
        print(f"  价量相关性因子计算耗时: {corr_end - corr_start}")
        # --- 计时结束：价量相关性因子 ---

        # --- 计时开始：流动性因子 ---
        liquidity_start = datetime.datetime.now()
        # 流动性缺口因子
        df_range = df_close_price.rolling(windows_1d).max() - df_close_price.rolling(windows_1d).min()
        # 流动性缺口因子
        df_liquidity_gap = df_range.div(df_volume + 1e-6)  # 使用 div
        # zscore 计算中，分母是 std，通常不为0，但为了安全也可以加 small number
        df_gap_zscore = (df_liquidity_gap - df_liquidity_gap.rolling(windows_20d).mean()).div(
            df_liquidity_gap.rolling(windows_20d).std() + 1e-6
        )
        liquidity_end = datetime.datetime.now()
        print(f"  流动性因子计算耗时: {liquidity_end - liquidity_start}")
        # --- 计时结束：流动性因子 ---

        # # --- 计时开始：价格位置因子 ---
        # price_position_start = datetime.datetime.now()
        # # 价格在近期高点的百分位（Price Percentile）
        # df_price_percentile = df_close_price.rolling(windows_20d).apply(
        #     lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
        # )
        # price_position_end = datetime.datetime.now()
        # print(f"  价格位置因子计算耗时: {price_position_end - price_position_start}")
        # --- 计时结束：价格位置因子 ---

        # --- 计时开始：其他直接因子 ---
        other_simple_start = datetime.datetime.now()
        # OBV 变化率（OBV Momentum）
        df_obv_change = df_obv.div(df_obv.shift(windows_1d) + 1e-6).sub(1).replace([np.inf, -np.inf], np.nan).fillna(0)
        # MFI 资金流量指数（Money Flow Index）
        df_typical_price = (df_high_price + df_low_price + df_close_price) / 3
        df_money_flow = df_typical_price * df_volume
        df_positive_flow = ((df_typical_price > df_typical_price.shift(1)) * df_money_flow).rolling(windows_14d).sum()
        df_negative_flow = ((df_typical_price < df_typical_price.shift(1)) * df_money_flow).rolling(windows_14d).sum()
        df_mfi = 100 - (100 / (1 + (df_positive_flow / (df_negative_flow + 1e-6))))
        # ATR 变化率（ATR Momentum）
        df_atr_change = df_atr.div(df_atr.shift(windows_1d) + 1e-6).sub(1).replace([np.inf, -np.inf], np.nan).fillna(0)
        # 波动率比率：短期/长期波动率（Volatility Ratio）
        df_vol_short = df_15min_rtn.rolling(4 * 24).std(ddof=1)  # 1天波动率
        df_vol_long = df_15min_rtn.rolling(windows_20d).std(ddof=1)  # 20天波动率
        df_vol_ratio = df_vol_short.div(df_vol_long + 1e-6)
        # 波动率聚集效应（Volatility Clustering）
        df_vol_cluster = (df_15min_rtn ** 2).rolling(windows_7d).mean()
        other_simple_end = datetime.datetime.now()
        print(f"  其他直接因子计算耗时: {other_simple_end - other_simple_start}")
        # --- 计时结束：其他直接因子 ---

        # --- 计时开始：交互项因子 ---
        interaction_start = datetime.datetime.now()
        # 动量与波动率交互项（Momentum-Vol Interaction）
        df_mom_vol_interact = df_7d_momentum * df_7d_volatility
        # 买入压力与RSI交互项
        df_buy_pressure_rsi_interact = df_buy_pressure * df_rsi
        # 价格偏离VWAP的Z-Score
        df_vwap_deviation_zscore = (df_vwap_deviation - df_vwap_deviation.rolling(windows_20d).mean()).div(
            df_vwap_deviation.rolling(windows_20d).std() + 1e-6
        )
        interaction_end = datetime.datetime.now()
        print(f"  交互项因子计算耗时: {interaction_end - interaction_start}")
        # --- 计时结束：交互项因子 ---

        # --- 计时开始：K线形态因子 ---
        pattern_start = datetime.datetime.now()
        # 锤子线形态识别（Hammer Indicator）
        df_hammer = ((lowershadow > 2 * uppershadow) & (df_close_price > df_open_price)).astype(int)
        # 十字星识别（Doji）
        df_doji = (np.abs(df_close_price - df_open_price) < 0.1 * price_range).astype(int)
        # 3.1 吞没形态（Engulfing Pattern）
        df_engulfing_bull = ((df_close_price.shift(1) < df_open_price.shift(1)) &
                             (df_close_price > df_open_price) &
                             (df_open_price < df_close_price.shift(1)) &
                             (df_close_price > df_open_price.shift(1))).astype(int)
        df_engulfing_bear = ((df_close_price.shift(1) > df_open_price.shift(1)) &
                             (df_close_price < df_open_price) &
                             (df_open_price > df_close_price.shift(1)) &
                             (df_close_price < df_open_price.shift(1))).astype(int)
        # 3.2 三连阳 / 三连阴
        df_up3 = ((df_close_price > df_open_price) &
                  (df_close_price.shift(1) > df_open_price.shift(1)) &
                  (df_close_price.shift(2) > df_open_price.shift(2))).astype(int)
        df_down3 = ((df_close_price < df_open_price) &
                    (df_close_price.shift(1) < df_open_price.shift(1)) &
                    (df_close_price.shift(2) < df_open_price.shift(2))).astype(int)
        # 3.3 乌云盖顶（Dark Cloud Cover）
        df_dark_cloud = ((df_close_price.shift(1) > df_open_price.shift(1)) &
                         (df_close_price < df_open_price) &
                         (df_open_price > df_close_price.shift(1)) &
                         (df_close_price < (df_open_price.shift(1) + df_close_price.shift(1)) / 2)).astype(int)
        pattern_end = datetime.datetime.now()
        print(f"  K线形态因子计算耗时: {pattern_end - pattern_start}")
        # --- 计时结束：K线形态因子 ---

        # --- 计时开始：市场状态与时间因子 ---
        regime_time_start = datetime.datetime.now()
        # # 市场状态分类（Regime Detection）—— one-hot 编码为3个布尔特征
        # df_vol_z = (df_7d_volatility - df_7d_volatility.rolling(windows_20d).mean()) / (
        #         df_7d_volatility.rolling(windows_20d).std() + 1e-6)
        # df_mom_abs_z = np.abs(df_7d_momentum).rolling(windows_20d).apply(lambda x: x.std() if len(x) > 1 else 0, raw=True) # 替代 zscore 方法
        # df_regime_trending = ((df_vol_z > 1) & (df_mom_abs_z > 1)).astype(int)  # 高波动 + 强动量
        # df_regime_choppy = ((df_vol_z > 1) & (df_mom_abs_z < 0.5)).astype(int)  # 高波动 + 弱动量
        # df_regime_lowvol = ((df_vol_z < -1) & (df_mom_abs_z < 0.5)).astype(int)  # 低波动 + 弱动量
        # 时间周期特征（如果 time_arr 是 DatetimeIndex）
        if isinstance(time_arr, (pd.DatetimeIndex, pd.Series)):
            t_index = pd.to_datetime(time_arr)
            df_hour_sin = np.sin(2 * np.pi * t_index.hour / 24)
            df_hour_cos = np.cos(2 * np.pi * t_index.hour / 24)
            # 广播到所有资产
            df_hour_sin = pd.DataFrame([df_hour_sin] * len(all_symbol_list)).T
            df_hour_sin.columns = all_symbol_list
            df_hour_sin.index = time_arr
            df_hour_cos = pd.DataFrame([df_hour_cos] * len(all_symbol_list)).T
            df_hour_cos.columns = all_symbol_list
            df_hour_cos.index = time_arr
        else:
            df_hour_sin = pd.DataFrame(0, index=time_arr, columns=all_symbol_list)
            df_hour_cos = pd.DataFrame(0, index=time_arr, columns=all_symbol_list)
        regime_time_end = datetime.datetime.now()
        print(f"  市场状态与时间因子计算耗时: {regime_time_end - regime_time_start}")
        # --- 计时结束：市场状态与时间因子 ---

        # # --- 计时开始：多尺度因子 ---
        # multi_scale_start = datetime.datetime.now()
        # # 1.1 多尺度动量差分（Momentum Divergence）
        # df_1d_momentum = df_vwap.div(df_vwap.shift(4 * 24)).sub(1).replace([np.inf, -np.inf], np.nan).fillna(0)
        # df_3d_momentum = df_vwap.div(df_vwap.shift(4 * 24 * 3)).sub(1).replace([np.inf, -np.inf], np.nan).fillna(0)
        # df_mom_divergence = df_1d_momentum - df_3d_momentum  # 短期加速 vs 长期趋势
        # # 1.2 波动率层级比（Volatility Hierarchy Ratio）
        # df_vol_1d = df_15min_rtn.rolling(4 * 24).std(ddof=1)
        # df_vol_3d = df_15min_rtn.rolling(4 * 24 * 3).std(ddof=1)
        # df_vol_hierarchy = df_vol_1d.div(df_vol_3d + 1e-6)
        # # 1.3 成交量趋势强度（Volume Trend Strength）
        # df_volume_trend = df_volume.rolling(windows_7d).apply(
        #     lambda x: np.polyfit(np.arange(len(x)), x, 1)[0] if len(x) > 1 else 0, raw=True
        # )
        # multi_scale_end = datetime.datetime.now()
        # print(f"  多尺度因子计算耗时: {multi_scale_end - multi_scale_start}")
        # --- 计时结束：多尺度因子 ---

        # # --- 计时开始：异常检测因子 ---
        # anomaly_start = datetime.datetime.now()
        # # 4.1 价格Z-Score（异常偏离）
        # df_price_z = (df_close_price - df_close_price.rolling(windows_20d).mean()).div(
        #     df_close_price.rolling(windows_20d).std() + 1e-6
        # )
        # # 4.2 成交量Z-Score（放量异常）
        # df_volume_z = (df_volume - df_volume.rolling(windows_20d).mean()).div(
        #     df_volume.rolling(windows_20d).std() + 1e-6
        # )
        # # 4.3 极端波动事件标记
        # df_extreme_vol = (df_7d_volatility > df_7d_volatility.rolling(windows_20d).quantile(0.95)).astype(int)
        # # 4.4 流动性枯竭检测
        # df_liquidity_dry = (df_volume < df_volume.rolling(windows_20d).quantile(0.05)).astype(int)
        # anomaly_end = datetime.datetime.now()
        # print(f"  异常检测因子计算耗时: {anomaly_end - anomaly_start}")
        # --- 计时结束：异常检测因子 ---

        # # --- 计时开始：横截面因子 ---
        # cross_sectional_start = datetime.datetime.now()
        # # 5.1 横截面动量（Cross-Sectional Momentum）
        # df_cs_momentum = df_7d_momentum.sub(df_7d_momentum.mean(axis=1), axis=0)
        # # 5.2 横截面波动率偏离
        # df_cs_vol_rank = df_7d_volatility.rank(axis=1, pct=True)
        # # 5.3 横截面买压强度
        # df_cs_buy_pressure_rank = df_buy_pressure.rank(axis=1, pct=True)
        # # 5.4 横截面RSI极值
        # df_rsi_z = (df_rsi - df_rsi.mean(axis=1).replace([np.inf, -np.inf], np.nan).fillna(50)) / \
        #            (df_rsi.std(axis=1).replace([np.inf, -np.inf], np.nan).fillna(1) + 1e-6)
        # cross_sectional_end = datetime.datetime.now()
        # print(f"  横截面因子计算耗时: {cross_sectional_end - cross_sectional_start}")
        # # --- 计时结束：横截面因子 ---

        # # --- 计时开始：流动性与成本因子 ---
        # liquidity_cost_start = datetime.datetime.now()
        # # 7.1 买卖价差代理（Bid-Ask Spread Proxy）
        # df_spread_proxy = (df_high_price - df_low_price).div(df_vwap + 1e-6)
        # # 7.2 冲击成本估计（Impact Cost Proxy）
        # df_impact_cost = (df_spread_proxy * df_volume).rolling(windows_1d).mean()
        # # 7.3 深度不足检测（Low Liquidity Warning）
        # df_low_liquidity = (df_volume < df_volume.rolling(windows_20d).quantile(0.1)).astype(int)
        # liquidity_cost_end = datetime.datetime.now()
        # print(f"  流动性与成本因子计算耗时: {liquidity_cost_end - liquidity_cost_start}")
        # # --- 计时结束：流动性与成本因子 ---

        print("  特征计算完成")
        # --- 修改点5: 调整传递给 train 的参数 ---
        # 调用train方法，使用滞后的7天24小时收益率作为目标值，调整后的因子作为输入进行模型训练
        # 注意：传递的因子数量和顺序必须与 train 方法的参数列表一致 (现在是47个因子)
        print("  开始调用 train 方法...")
        self.train(df_24hour_rtn.shift(-windows_1d),
                   df_7d_volatility, df_7d_momentum, df_amount_sum,
                   df_vol_momentum, df_atr, df_macd, df_buy_pressure,
                   df_bb_position, df_obv_ratio, df_obv_zscore,
                   df_corr_diff, df_corr_extreme, df_gap_zscore,
                   df_rsi, df_buy_ratio, df_vwap_deviation, df_4h_momentum,
                   price_range, change_rate, averagetradingprice,
                   uppershadow, lowershadow, buyvolumnratio,
                   buyamountratio, sellvolume, BSratio,
                   # df_price_percentile,
                   # df_volume_skew, # 已注释
                   df_obv_change, df_mfi, df_atr_change,
                   df_vol_ratio, df_vol_cluster, df_mom_vol_interact,
                   df_buy_pressure_rsi_interact, df_vwap_deviation_zscore,
                   df_hammer, df_doji,
                   # df_trend_strength, # 已注释
                   # df_regime_trending, df_regime_choppy,
                   # df_regime_lowvol,
                   df_hour_sin, df_hour_cos,
                   # df_mom_divergence, df_vol_hierarchy,
                   # df_volume_kurtosis, df_return_entropy, # 已注释
                   # df_price_skew, df_fractal_dim, df_hurst, # 已注释
                   # df_engulfing_bull, df_engulfing_bear,
                   # df_up3, df_down3, df_price_z, df_volume_z,
                   # df_cs_momentum,
                   # df_autocorr_lag, df_cycle_strength, # 已注释
                   # df_spread_proxy, df_impact_cost
                   )
        # --- 修改点结束 ---
        run_end_time = datetime.datetime.now()
        print(f"--- run 方法结束，总耗时: {run_end_time - run_start_time} ---")


if __name__ == '__main__':
    model = OlsModel()
    model.run()