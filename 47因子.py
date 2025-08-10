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
        with mp.Pool(mp.cpu_count() - 2) as pool:
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

    # def train(self, df_target, df_factor1, df_factor2, df_factor3, df_factor4, df_factor5, df_factor6, df_factor7,
    #       df_factor8, df_factor9, df_factor10, df_factor11, df_factor12, df_factor13, df_factor14, df_factor15,
    #       df_factor16, df_factor17):

    #     # 将因子数据转换为长格式
    #     factor1_long = df_factor1.stack()
    #     factor2_long = df_factor2.stack()
    #     factor3_long = df_factor3.stack()
    #     factor4_long = df_factor4.stack()
    #     factor5_long = df_factor5.stack()
    #     factor6_long = df_factor6.stack()
    #     factor7_long = df_factor7.stack()
    #     factor8_long = df_factor8.stack()
    #     factor9_long = df_factor9.stack()
    #     factor10_long = df_factor10.stack()
    #     factor11_long = df_factor11.stack()
    #     factor12_long = df_factor12.stack()
    #     factor13_long = df_factor13.stack()
    #     factor14_long = df_factor14.stack()
    #     factor15_long = df_factor15.stack()
    #     factor16_long = df_factor16.stack()
    #     factor17_long = df_factor17.stack()
    #     target_long = df_target.stack()

    #     # 设置列名
    #     factor_names = [f'factor{i}' for i in range(1, 18)]
    #     factors_long = [factor1_long, factor2_long, factor3_long, factor4_long, factor5_long,
    #                    factor6_long, factor7_long, factor8_long, factor9_long, factor10_long,
    #                    factor11_long, factor12_long, factor13_long, factor14_long,
    #                    factor15_long, factor16_long, factor17_long]

    #     for i, factor in enumerate(factors_long):
    #         factor.name = factor_names[i]
    #     target_long.name = 'target'

    #     # 合并数据
    #     data = pd.concat(factors_long + [target_long], axis=1)
    #     data = data.dropna()

    #     # 限制序列长度和数据量
    #     seq_length = 24 * 4 * 3  # 减少到3天，原来是5天
    #     min_sequence_length = seq_length + 10  # 最小序列长度

    #     # 按symbol分组构建时序序列（优化版本）
    #     def create_symbol_sequences_optimized(grouped_data, seq_length, max_sequences_per_symbol=1000):
    #         X_seq_list, y_seq_list = [], []

    #         # 按symbol分组
    #         grouped = grouped_data.groupby(level=1)

    #         total_sequences = 0
    #         max_total_sequences = 500000  # 限制总序列数

    #         for symbol, symbol_data in grouped:
    #             if len(symbol_data) > min_sequence_length and total_sequences < max_total_sequences:
    #                 # 提取因子和目标值
    #                 factor_values = symbol_data[factor_names].values
    #                 target_values = symbol_data['target'].values

    #                 # 限制每个symbol的序列数量
    #                 n_sequences = min(len(factor_values) - seq_length, max_sequences_per_symbol)

    #                 # 构建滑动窗口序列
    #                 for i in range(n_sequences):
    #                     if total_sequences >= max_total_sequences:
    #                         break
    #                     X_seq_list.append(factor_values[i:(i + seq_length)])      # [seq_length, 17]
    #                     y_seq_list.append(target_values[i + seq_length - 1])      # 标量
    #                     total_sequences += 1

    #                 if total_sequences >= max_total_sequences:
    #                     print(f"达到最大序列数限制: {max_total_sequences}")
    #                     break

    #         print(f"总共创建了 {total_sequences} 个序列")
    #         return np.array(X_seq_list), np.array(y_seq_list)

    #     # 构建时序序列
    #     print("开始构建时序序列...")
    #     X_sequences, y_sequences = create_symbol_sequences_optimized(data, seq_length)

    #     if len(X_sequences) == 0:
    #         print("没有足够的数据构建序列")
    #         return

    #     print(f"构建的序列形状: X={X_sequences.shape}, y={y_sequences.shape}")

    #     # 数据标准化（使用32位浮点数节省内存）
    #     print("开始数据标准化...")
    #     scaler_X = StandardScaler()
    #     scaler_y = StandardScaler()

    #     # 分批处理以节省内存
    #     batch_size = 10000
    #     X_scaled_list = []

    #     for i in range(0, len(X_sequences), batch_size):
    #         batch = X_sequences[i:i+batch_size]
    #         # 重塑批次用于标准化
    #         batch_reshaped = batch.reshape(-1, batch.shape[2]).astype(np.float32)
    #         batch_scaled = scaler_X.transform(batch_reshaped) if hasattr(scaler_X, 'scale_') else scaler_X.fit_transform(batch_reshaped)
    #         # 恢复形状
    #         batch_restored = batch_scaled.reshape(batch.shape)
    #         X_scaled_list.append(batch_restored)

    #     X_scaled = np.concatenate(X_scaled_list, axis=0)
    #     y_scaled = scaler_y.fit_transform(y_sequences.reshape(-1, 1)).astype(np.float32).flatten()

    #     print("数据标准化完成")

    #     # 清理内存
    #     del X_sequences, y_sequences, X_scaled_list
    #     import gc
    #     gc.collect()

    #     # 转换为PyTorch张量
    #     print("转换为PyTorch张量...")
    #     X_tensor = torch.FloatTensor(X_scaled)
    #     y_tensor = torch.FloatTensor(y_scaled)

    #     # 检查是否有GPU可用
    #     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #     print(f"Using device: {device}")

    #     # 创建数据集和数据加载器
    #     dataset = TensorDataset(X_tensor, y_tensor)
    #     dataloader = DataLoader(dataset, batch_size=32, shuffle=True)  # 减小batch_size

    #     # 初始化模型
    #     model = LSTMModel(input_size=17, hidden_size=64, num_layers=2, output_size=1, dropout=0.2).to(device)

    #     # 定义损失函数和优化器
    #     criterion = nn.MSELoss()
    #     optimizer = optim.Adam(model.parameters(), lr=0.001)

    #     # 学习率调度器
    #     scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)

    #     # 训练模型
    #     model.train()
    #     num_epochs = 50  # 减少epoch数
    #     for epoch in range(num_epochs):
    #         total_loss = 0
    #         for batch_X, batch_y in dataloader:
    #             batch_X, batch_y = batch_X.to(device), batch_y.to(device)

    #             # 前向传播
    #             outputs = model(batch_X)
    #             loss = criterion(outputs.squeeze(), batch_y)

    #             # 反向传播和优化
    #             optimizer.zero_grad()
    #             loss.backward()
    #             torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    #             optimizer.step()

    #             total_loss += loss.item()

    #         scheduler.step()  # 更新学习率

    #         if (epoch + 1) % 10 == 0:
    #             print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss / len(dataloader):.6f}, LR: {scheduler.get_last_lr()[0]:.6f}')

    #     # 使用训练好的模型进行预测（分批预测以节省内存）
    #     print("开始预测...")
    #     model.eval()
    #     y_pred_list = []

    #     with torch.no_grad():
    #         for i in range(0, len(X_tensor), 1000):  # 分批预测
    #             batch_X = X_tensor[i:i+1000].to(device)
    #             batch_pred = model(batch_X).cpu().numpy().flatten()
    #             y_pred_list.append(batch_pred)

    #     y_pred_scaled = np.concatenate(y_pred_list)
    #     y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()

    #     print("预测完成")

    #     # 清理GPU内存
    #     del X_tensor, y_tensor, dataset, dataloader
    #     torch.cuda.empty_cache()

    #     # 后续处理代码...（保持原有逻辑）
    #     # 注意：由于我们限制了序列数量，预测结果的处理需要相应调整
    #     # 重建预测结果DataFrame
    #     # 为每个序列创建对应的datetime和symbol
    #     pred_data_list = []
    #     seq_idx = 0

    #     grouped = data.groupby(level=1)
    #     for symbol, symbol_data in grouped:
    #         if len(symbol_data) > seq_length:
    #             # 为每个序列创建预测结果
    #             for i in range(min(len(symbol_data) - seq_length, 1000)):  # 限制每个symbol的序列数
    #                 if seq_idx < len(y_pred):
    #                     # 预测的是第 i+seq_length-1 个时间点
    #                     pred_datetime = symbol_data.index[i + seq_length - 1][0]  # 获取datetime
    #                     pred_data_list.append({
    #                         'datetime': pred_datetime,
    #                         'symbol': symbol,
    #                         'y_pred': y_pred[seq_idx],
    #                         'target': symbol_data['target'].iloc[i + seq_length - 1]
    #                     })
    #                     seq_idx += 1

    #     # 创建预测结果DataFrame
    #     pred_df = pd.DataFrame(pred_data_list)
    #     pred_df['id'] = pred_df['datetime'].astype(str) + "_" + pred_df['symbol']
    #     pred_df = pred_df.set_index(['datetime', 'symbol'])

    #     # 将预测结果合并回原始数据
    #     data_with_pred = data.copy()
    #     data_with_pred['y_pred'] = np.nan  # 初始化预测列

    #     # 手动将预测结果填入对应位置
    #     for idx, row in pred_df.iterrows():
    #         if idx in data_with_pred.index:
    #             data_with_pred.loc[idx, 'y_pred'] = row['y_pred']

    #     data_with_pred = data_with_pred.dropna()

    #     # 后续处理保持完全不变
    #     df_submit = data_with_pred.reset_index(level=0)
    #     df_submit = df_submit[['level_0', 'y_pred']]
    #     df_submit['symbol'] = df_submit.index.values
    #     df_submit = df_submit[['level_0', 'symbol', 'y_pred']]
    #     df_submit.columns = ['datetime', 'symbol', 'predict_return']
    #     df_submit = df_submit[df_submit['datetime'] >= self.start_datetime]
    #     df_submit["id"] = df_submit["datetime"].astype(str) + "_" + df_submit["symbol"]
    #     df_submit = df_submit[['id', 'predict_return']]

    #     print(df_submit)

    #     df_submission_id = pd.read_csv("submission_id.csv")
    #     id_list = df_submission_id["id"].tolist()
    #     df_submit_competion = df_submit[df_submit['id'].isin(id_list)]
    #     missing_elements = list(set(id_list) - set(df_submit_competion['id']))
    #     new_rows = pd.DataFrame({'id': missing_elements, 'predict_return': [0] * len(missing_elements)})
    #     df_submit_competion = pd.concat([df_submit, new_rows], ignore_index=True)
    #     print(df_submit_competion.shape)
    #     df_submit_competion.to_csv("submit.csv", index=False)

    #     df_check = data_with_pred.reset_index(level=0)
    #     df_check = df_check[['level_0', 'target']]
    #     df_check['symbol'] = df_check.index.values
    #     df_check = df_check[['level_0', 'symbol', 'target']]
    #     df_check.columns = ['datetime', 'symbol', 'true_return']
    #     df_check = df_check[df_check['datetime'] >= self.start_datetime]
    #     df_check["id"] = df_check["datetime"].astype(str) + "_" + df_check["symbol"]
    #     df_check = df_check[['id', 'true_return']]

    #     print(df_check)

    #     df_check.to_csv("check.csv", index=False)

    #     # 在整个样本上计算加权Spearman相关系数
    #     if len(data_with_pred) > 0:
    #         rho_overall = self.weighted_spearmanr(data_with_pred['target'].dropna(),
    #                                               data_with_pred['y_pred'].dropna())
    #         print(f"加权Spearman相关系数: {rho_overall:.4f}")

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
        price_range = df_high_price-df_close_price
        #价格变化率
        change_rate = (df_close_price-df_open_price)/df_close_price
        #每笔交易的平均金额
        averagetradingprice = df_buy_volume/df_amount
        #上影线长度
        uppershadow = df_high_price - np.maximum(df_open_price , df_close_price)
        #下影线长度
        lowershadow = np.minimum(df_open_price,df_close_price) - df_low_price
        # 买方交易量占比
        buyvolumnratio = df_buy_volume/df_volume
        # 买方交易额占比
        buyamountratio = df_buy_volume/df_amount
        # 卖方交易量
        sellvolume = df_volume - df_buy_volume
        # 买卖交易量比
        BSratio = df_buy_volume/sellvolume
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

        # --- 计时开始：价格位置因子 ---
        price_position_start = datetime.datetime.now()
        # 价格在近期高点的百分位（Price Percentile）
        df_price_percentile = df_close_price.rolling(windows_20d).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
        )
        price_position_end = datetime.datetime.now()
        print(f"  价格位置因子计算耗时: {price_position_end - price_position_start}")
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
        # 市场状态分类（Regime Detection）—— one-hot 编码为3个布尔特征
        df_vol_z = (df_7d_volatility - df_7d_volatility.rolling(windows_20d).mean()) / (
                df_7d_volatility.rolling(windows_20d).std() + 1e-6)
        df_mom_abs_z = np.abs(df_7d_momentum).rolling(windows_20d).apply(lambda x: x.std() if len(x) > 1 else 0, raw=True) # 替代 zscore 方法
        df_regime_trending = ((df_vol_z > 1) & (df_mom_abs_z > 1)).astype(int)  # 高波动 + 强动量
        df_regime_choppy = ((df_vol_z > 1) & (df_mom_abs_z < 0.5)).astype(int)  # 高波动 + 弱动量
        df_regime_lowvol = ((df_vol_z < -1) & (df_mom_abs_z < 0.5)).astype(int)  # 低波动 + 弱动量
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

        # --- 计时开始：多尺度因子 ---
        multi_scale_start = datetime.datetime.now()
        # 1.1 多尺度动量差分（Momentum Divergence）
        df_1d_momentum = df_vwap.div(df_vwap.shift(4 * 24)).sub(1).replace([np.inf, -np.inf], np.nan).fillna(0)
        df_3d_momentum = df_vwap.div(df_vwap.shift(4 * 24 * 3)).sub(1).replace([np.inf, -np.inf], np.nan).fillna(0)
        df_mom_divergence = df_1d_momentum - df_3d_momentum  # 短期加速 vs 长期趋势
        # 1.2 波动率层级比（Volatility Hierarchy Ratio）
        df_vol_1d = df_15min_rtn.rolling(4 * 24).std(ddof=1)
        df_vol_3d = df_15min_rtn.rolling(4 * 24 * 3).std(ddof=1)
        df_vol_hierarchy = df_vol_1d.div(df_vol_3d + 1e-6)
        # 1.3 成交量趋势强度（Volume Trend Strength）
        df_volume_trend = df_volume.rolling(windows_7d).apply(
            lambda x: np.polyfit(np.arange(len(x)), x, 1)[0] if len(x) > 1 else 0, raw=True
        )
        multi_scale_end = datetime.datetime.now()
        print(f"  多尺度因子计算耗时: {multi_scale_end - multi_scale_start}")
        # --- 计时结束：多尺度因子 ---

        # --- 计时开始：异常检测因子 ---
        anomaly_start = datetime.datetime.now()
        # 4.1 价格Z-Score（异常偏离）
        df_price_z = (df_close_price - df_close_price.rolling(windows_20d).mean()).div(
            df_close_price.rolling(windows_20d).std() + 1e-6
        )
        # 4.2 成交量Z-Score（放量异常）
        df_volume_z = (df_volume - df_volume.rolling(windows_20d).mean()).div(
            df_volume.rolling(windows_20d).std() + 1e-6
        )
        # 4.3 极端波动事件标记
        df_extreme_vol = (df_7d_volatility > df_7d_volatility.rolling(windows_20d).quantile(0.95)).astype(int)
        # 4.4 流动性枯竭检测
        df_liquidity_dry = (df_volume < df_volume.rolling(windows_20d).quantile(0.05)).astype(int)
        anomaly_end = datetime.datetime.now()
        print(f"  异常检测因子计算耗时: {anomaly_end - anomaly_start}")
        # --- 计时结束：异常检测因子 ---

        # --- 计时开始：横截面因子 ---
        cross_sectional_start = datetime.datetime.now()
        # 5.1 横截面动量（Cross-Sectional Momentum）
        df_cs_momentum = df_7d_momentum.sub(df_7d_momentum.mean(axis=1), axis=0)
        # 5.2 横截面波动率偏离
        df_cs_vol_rank = df_7d_volatility.rank(axis=1, pct=True)
        # 5.3 横截面买压强度
        df_cs_buy_pressure_rank = df_buy_pressure.rank(axis=1, pct=True)
        # 5.4 横截面RSI极值
        df_rsi_z = (df_rsi - df_rsi.mean(axis=1).replace([np.inf, -np.inf], np.nan).fillna(50)) / \
                   (df_rsi.std(axis=1).replace([np.inf, -np.inf], np.nan).fillna(1) + 1e-6)
        cross_sectional_end = datetime.datetime.now()
        print(f"  横截面因子计算耗时: {cross_sectional_end - cross_sectional_start}")
        # --- 计时结束：横截面因子 ---

        # --- 计时开始：流动性与成本因子 ---
        liquidity_cost_start = datetime.datetime.now()
        # 7.1 买卖价差代理（Bid-Ask Spread Proxy）
        df_spread_proxy = (df_high_price - df_low_price).div(df_vwap + 1e-6)
        # 7.2 冲击成本估计（Impact Cost Proxy）
        df_impact_cost = (df_spread_proxy * df_volume).rolling(windows_1d).mean()
        # 7.3 深度不足检测（Low Liquidity Warning）
        df_low_liquidity = (df_volume < df_volume.rolling(windows_20d).quantile(0.1)).astype(int)
        liquidity_cost_end = datetime.datetime.now()
        print(f"  流动性与成本因子计算耗时: {liquidity_cost_end - liquidity_cost_start}")
        # --- 计时结束：流动性与成本因子 ---

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
                   df_price_percentile,
                   # df_volume_skew, # 已注释
                   df_obv_change, df_mfi, df_atr_change,
                   df_vol_ratio, df_vol_cluster, df_mom_vol_interact,
                   df_buy_pressure_rsi_interact, df_vwap_deviation_zscore,
                   df_hammer, df_doji,
                   # df_trend_strength, # 已注释
                   df_regime_trending, df_regime_choppy,
                   df_regime_lowvol, df_hour_sin, df_hour_cos,
                   df_mom_divergence, df_vol_hierarchy,
                   # df_volume_kurtosis, df_return_entropy, # 已注释
                   # df_price_skew, df_fractal_dim, df_hurst, # 已注释
                   df_engulfing_bull, df_engulfing_bear,
                   df_up3, df_down3, df_price_z, df_volume_z,
                   df_cs_momentum,
                   # df_autocorr_lag, df_cycle_strength, # 已注释
                   df_spread_proxy, df_impact_cost
                   )
        # --- 修改点结束 ---
        run_end_time = datetime.datetime.now()
        print(f"--- run 方法结束，总耗时: {run_end_time - run_start_time} ---")


if __name__ == '__main__':
    model = OlsModel()
    model.run()