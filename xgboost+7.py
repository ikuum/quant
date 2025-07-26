import numpy as np
import pandas as pd
import datetime, os, time
import multiprocessing as mp  # 用于并行处理
import torch  # PyTorch深度学习框架
import xgboost as xgb  # XGBoost梯度提升树库
from sklearn.preprocessing import StandardScaler  # 数据标准化
from sklearn.model_selection import TimeSeriesSplit  # 时间序列交叉验证
import shap  # SHAP值解释库


class OptimizedModel:
    def __init__(self):
        # 初始化模型参数
        self.train_data_path = "train_data"  # 训练数据目录
        self.submission_id_path = "submission_id.csv"  # 提交ID文件路径
        self.start_datetime = datetime.datetime(2021, 3, 1, 0, 0, 0)  # 有效数据起始时间
        self.scaler = StandardScaler()  # 数据标准化器
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'  # 自动选择GPU或CPU
        print(f"Using device: {self.device}")  # 打印使用的设备

    def get_all_symbol_list(self):
        # 获取所有加密货币符号列表
        try:
            # 列出训练数据目录中的所有文件
            parquet_name_list = os.listdir(self.train_data_path)
            # 提取文件名中的币种符号（去除扩展名）
            symbol_list = [parquet_name.split(".")[0] for parquet_name in parquet_name_list]
            return symbol_list
        except Exception as e:
            print(f"Error in get_all_symbol_list: {e}")
            return []  # 出错时返回空列表

    def compute_factors_torch(self, df):
        # 使用PyTorch计算技术指标因子（在GPU上加速）

        # 将DataFrame列转换为PyTorch张量并移到指定设备
        close = torch.tensor(df['close_price'].values, dtype=torch.float32, device=self.device)
        volume = torch.tensor(df['volume'].values, dtype=torch.float32, device=self.device)
        amount = torch.tensor(df['amount'].values, dtype=torch.float32, device=self.device)
        high = torch.tensor(df['high_price'].values, dtype=torch.float32, device=self.device)
        low = torch.tensor(df['low_price'].values, dtype=torch.float32, device=self.device)
        buy_volume = torch.tensor(df['buy_volume'].values, dtype=torch.float32, device=self.device)

        # 计算成交量加权平均价(VWAP)
        vwap = torch.where(volume > 0, amount / volume, close)
        vwap = torch.where(torch.isfinite(vwap), vwap, close)  # 处理无穷值

        # 计算相对强弱指数(RSI)
        delta = torch.diff(close, prepend=close[:1])  # 计算价格变化
        gain = torch.where(delta > 0, delta, torch.tensor(0.0, device=self.device))  # 正变化
        loss = torch.where(delta < 0, -delta, torch.tensor(0.0, device=self.device))  # 负变化

        # 初始化平均增益和平均损失
        init_gain = gain[:14].mean() if len(gain) > 14 else torch.tensor(0.0, device=self.device)
        init_loss = loss[:14].mean() if len(gain) > 14 else torch.tensor(0.0, device=self.device)

        # 创建存储平均增益和损失的张量
        avg_gain = torch.zeros_like(close, device=self.device)
        avg_loss = torch.zeros_like(close, device=self.device)
        avg_gain[:14] = init_gain
        avg_loss[:14] = init_loss

        # 迭代计算平均增益和损失
        for i in range(14, len(close)):
            avg_gain[i] = (avg_gain[i - 1] * 13 + gain[i]) / 14
            avg_loss[i] = (avg_loss[i - 1] * 13 + loss[i]) / 14

        # 计算相对强度(RS)和RSI
        rs = torch.where(avg_loss > 0, avg_gain / avg_loss, torch.tensor(0.0, device=self.device))
        rsi = 100 - 100 / (1 + rs)
        rsi = torch.where(torch.isnan(rsi), torch.tensor(50.0, device=self.device), rsi)  # 处理NaN

        # 计算MACD指标
        ema12 = torch.zeros_like(close, device=self.device)  # 12日EMA
        ema26 = torch.zeros_like(close, device=self.device)  # 26日EMA
        alpha12 = 2 / (12 + 1)  # EMA平滑系数
        alpha26 = 2 / (26 + 1)

        # 初始化EMA
        ema12[:12] = close[:12].mean()
        ema26[:26] = close[:26].mean()

        # 计算EMA
        for i in range(12, len(close)):
            ema12[i] = alpha12 * close[i] + (1 - alpha12) * ema12[i - 1]
        for i in range(26, len(close)):
            ema26[i] = alpha26 * close[i] + (1 - alpha26) * ema26[i - 1]

        # 计算MACD线
        macd = ema12 - ema26
        macd = torch.where(torch.isnan(macd), torch.tensor(0.0, device=self.device), macd)  # 处理NaN

        # 计算平均真实波幅(ATR)
        tr = torch.max(high - low, torch.max(torch.abs(high - close), torch.abs(low - close)))  # 真实波幅
        atr = torch.zeros_like(tr, device=self.device)  # 存储ATR

        # 计算ATR
        for i in range(14, len(tr)):
            atr[i] = (atr[i - 1] * 13 + tr[i]) / 14
        atr = torch.where(torch.isnan(atr), torch.tensor(0.0, device=self.device), atr)  # 处理NaN

        # 计算买入量比例
        buy_ratio = torch.where(volume > 0, buy_volume / volume, torch.tensor(0.5, device=self.device))

        # 计算VWAP偏离度
        vwap_deviation = (close - vwap) / torch.where(vwap != 0, vwap, torch.tensor(1.0, device=self.device))
        vwap_deviation = torch.where(torch.isfinite(vwap_deviation), vwap_deviation,
                                     torch.tensor(0.0, device=self.device))  # 处理无穷值

        #   将计算结果添加回DataFrame
        df['vwap'] = vwap.cpu().numpy()
        df['rsi'] = rsi.cpu().numpy()
        df['macd'] = macd.cpu().numpy()
        df['atr'] = atr.cpu().numpy()
        df['buy_ratio'] = buy_ratio.cpu().numpy()
        df['vwap_deviation'] = vwap_deviation.cpu().numpy()
        return df

    def get_single_symbol_kline_data(self, symbol):
        # 加载单个加密货币的K线数据
        try:
            # 读取Parquet文件
            df = pd.read_parquet(f"{self.train_data_path}/{symbol}.parquet")
            # 转换时间戳为datetime对象
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            # 设置时间戳为索引
            df = df.set_index('timestamp')
            # 转换为float64类型
            df = df.astype(np.float64)

            #  检查必需列是否存在
            required_cols = ['open_price', 'high_price', 'low_price', 'close_price', 'volume', 'amount', 'buy_volume']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                print(f"{symbol} missing columns: {missing_cols}")
                # 返回包含所有必需列的DataFrame
                return pd.DataFrame(
                    columns=required_cols + ['vwap', 'rsi', 'macd', 'buy_ratio', 'vwap_deviation', 'atr'])

            # 对价格和交易量进行截断处理（去除极端值）
            df['close_price'] = df['close_price'].clip(df['close_price'].quantile(0.01),
                                                       df['close_price'].quantile(0.99))
            df['volume'] = df['volume'].clip(df['volume'].quantile(0.01), df['volume'].quantile(0.99))

            # 计算技术指标
            df = self.compute_factors_torch(df)
            print(f"Loaded data for {symbol}, shape: {df.shape}, vwap NaNs: {df['vwap'].isna().sum()}")
            return df
        except Exception as e:
            print(f"Error loading {symbol}: {e}")
            # 出错时返回包含所有列的DataFrame
            return pd.DataFrame(
                columns=['open_price', 'high_price', 'low_price', 'close_price', 'volume', 'amount', 'buy_volume',
                         'vwap',
                         'rsi', 'macd', 'buy_ratio', 'vwap_deviation', 'atr'])

    def get_all_symbol_kline(self):
        # 加载所有加密货币的K线数据
        t0 = datetime.datetime.now()  # 记录开始时间

        # 创建进程池（使用2个工作进程）
        pool = mp.Pool(2)
        # 获取所有币种列表
        all_symbol_list = self.get_all_symbol_list()
        if not all_symbol_list:
            print("No symbols found, exiting.")
            pool.close()
            return [], [], [], [], [], [], [], [], [], []  # 返回空列表

        # 并行加载每个币种的数据
        df_list = [pool.apply_async(self.get_single_symbol_kline_data, (symbol,)) for symbol in all_symbol_list]
        pool.close()
        pool.join()  # 等待所有任务完成

        # 检查哪些币种成功加载
        loaded_symbols = []
        for i, s in zip(df_list, all_symbol_list):
            df = i.get()
            if not df.empty and 'vwap' in df.columns:
                loaded_symbols.append(s)
            else:
                print(f"{s} failed: empty or missing 'vwap'")
        failed_symbols = [s for s in all_symbol_list if s not in loaded_symbols]
        print(f"Failed symbols: {failed_symbols}")

        # 创建时间索引（从开始时间到2024年底，15分钟间隔）
        time_index = pd.date_range(start=self.start_datetime, end='2024-12-31', freq='15min')

        # 构建开盘价DataFrame
        df_open_price = pd.concat(
            [i.get()['open_price'] for i in df_list if not i.get().empty and 'open_price' in i.get().columns],
            axis=1).sort_index(ascending=True)
        print(f"df_open_price index dtype: {df_open_price.index.dtype}, shape: {df_open_price.shape}")
        df_open_price.columns = loaded_symbols
        # 重新索引以填充缺失的币种和时间点
        df_open_price = df_open_price.reindex(columns=all_symbol_list, fill_value=0).reindex(time_index, method='ffill')
        # 获取时间数组
        time_arr = pd.to_datetime(df_open_price.index).values

        # 辅助函数：对齐不同币种的数据
        def align_df(arr, valid_symbols, key):
            # 收集所有有效币种的数据
            valid_dfs = [df[key] for df, s in zip([i.get() for i in df_list], all_symbol_list) if
                         not df.empty and key in df.columns and s in valid_symbols]
            if not valid_dfs:
                print(f"No valid data for {key}, filling with zeros")
                return np.zeros((len(time_index), len(all_symbol_list)))
            # 合并数据
            df = pd.concat(valid_dfs, axis=1).sort_index(ascending=True)
            df.columns = valid_symbols
            # 重新索引以填充缺失的币种和时间点
            return df.reindex(columns=all_symbol_list, fill_value=0).reindex(time_index, method='ffill').values

        # 对齐各种指标数据
        vwap_arr = align_df(df_list, loaded_symbols, 'vwap')
        amount_arr = align_df(df_list, loaded_symbols, 'amount')
        atr_arr = align_df(df_list, loaded_symbols, 'atr')
        macd_arr = align_df(df_list, loaded_symbols, 'macd')
        buy_volume_arr = align_df(df_list, loaded_symbols, 'buy_volume')
        volume_arr = align_df(df_list, loaded_symbols, 'volume')

        # 打印耗时
        print(f"Finished get all symbols kline, time elapsed: {datetime.datetime.now() - t0}")
        # 返回所有数据
        return all_symbol_list, time_arr, vwap_arr, amount_arr, atr_arr, macd_arr, buy_volume_arr, volume_arr

    def weighted_spearmanr(self, y_true, y_pred):
        # 计算加权Spearman相关系数
        n = len(y_true)  # 样本数量
        # 计算真实值和预测值的排名（降序）
        r_true = pd.Series(y_true).rank(ascending=False, method='average')
        r_pred = pd.Series(y_pred).rank(ascending=False, method='average')
        # 归一化排名到[-1, 1]范围
        x = 2 * (r_true - 1) / (n - 1) - 1
        w = x ** 2  # 计算权重（排名靠前的样本权重更大）
        w_sum = w.sum()  # 权重总和
        # 计算加权平均值
        mu_true = (w * r_true).sum() / w_sum
        mu_pred = (w * r_pred).sum() / w_sum
        # 计算加权协方差
        cov = (w * (r_true - mu_true) * (r_pred - mu_pred)).sum()
        # 计算加权方差
        var_true = (w * (r_true - mu_true) ** 2).sum()
        var_pred = (w * (r_pred - mu_pred) ** 2).sum()
        # 返回相关系数（避免除以零）
        return cov / np.sqrt(var_true * var_pred) if var_true * var_pred > 0 else 0

    def train(self, df_target, df_4h_momentum, df_7d_momentum, df_amount_sum, df_vol_momentum, df_atr, df_macd,
              df_buy_pressure):
        # 训练XGBoost模型

        # 将因子数据转换为长格式（多索引：时间和符号）
        factor1_long = df_4h_momentum.stack()
        factor2_long = df_7d_momentum.stack()
        factor3_long = df_amount_sum.stack()
        factor4_long = df_vol_momentum.stack()
        factor5_long = df_atr.stack()
        factor6_long = df_macd.stack()
        factor7_long = df_buy_pressure.stack()
        target_long = df_target.stack()  # 目标变量（未来24小时收益率）

        # 设置列名
        factor1_long.name = '4h_momentum'
        factor2_long.name = '7d_momentum'
        factor3_long.name = 'amount_sum'
        factor4_long.name = 'vol_momentum'
        factor5_long.name = 'atr'
        factor6_long.name = 'macd'
        factor7_long.name = 'buy_pressure'
        target_long.name = 'target'

        # 合并所有因子和目标变量
        data = pd.concat(
            [factor1_long, factor2_long, factor3_long, factor4_long, factor5_long, factor6_long, factor7_long,
             target_long],
            axis=1)
        print(f"Data size before dropna: {len(data)}")
        # 处理无穷值和缺失值
        data = data.replace([np.inf, -np.inf], np.nan).dropna()
        print(f"Data size after dropna: {len(data)}")

        # 准备特征和目标变量
        X = data[['4h_momentum', '7d_momentum', 'amount_sum', 'vol_momentum', 'atr', 'macd', 'buy_pressure']]
        y = data['target'].replace([np.inf, -np.inf], 0)  # 处理目标变量的无穷值

        # 标准化特征
        X_scaled = self.scaler.fit_transform(X)

        # 时间序列交叉验证
        tscv = TimeSeriesSplit(n_splits=5)
        best_score = -np.inf  # 最佳分数初始化为负无穷
        best_model = None  # 最佳模型初始化为None

        # 交叉验证循环
        for train_idx, val_idx in tscv.split(X_scaled):
            # 划分训练集和验证集
            X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            y_train_clean = y_train.fillna(0)  # 处理缺失值

            # 创建样本权重（对极端值给予更高权重）
            sample_weight = np.where(
                (y_train_clean > y_train_clean.quantile(0.9)) | (y_train_clean < y_train_clean.quantile(0.1)), 2, 1)

            # 初始化XGBoost回归模型
            model = xgb.XGBRegressor(
                objective='reg:squarederror',  # 回归任务
                learning_rate=0.05,  # 学习率
                max_depth=6,  # 树的最大深度
                subsample=0.8,  # 样本采样比例
                n_estimators=200,  # 树的数量
                early_stopping_rounds=10,  # 早停轮数
                random_state=42  # 随机种子
            )
            # 训练模型
            model.fit(X_train, y_train, sample_weight=sample_weight, eval_set=[(X_val, y_val)], verbose=False)

            # 在验证集上预测
            y_pred_val = model.predict(X_val)
            # 计算加权Spearman相关系数
            score = self.weighted_spearmanr(y_val, y_pred_val)
            # 更新最佳模型
            if score > best_score:
                best_score = score
                best_model = model

        print(f"Best validation Spearman score: {best_score:.4f}")

        # 在整个数据集上预测
        data['y_pred'] = best_model.predict(X_scaled)
        # 处理预测值的无穷值和缺失值
        data['y_pred'] = data['y_pred'].replace([np.inf, -np.inf], 0).fillna(0)
        # 使用指数加权移动平均平滑预测值
        data['y_pred'] = data['y_pred'].ewm(span=5).mean()

        # 准备提交文件
        df_submit = data.reset_index(level=0)
        df_submit = df_submit[['level_0', 'y_pred']]
        df_submit['symbol'] = df_submit.index.values
        df_submit = df_submit[['level_0', 'symbol', 'y_pred']]
        df_submit.columns = ['datetime', 'symbol', 'predict_return']
        # 过滤起始时间之后的数据
        df_submit = df_submit[df_submit['datetime'] >= self.start_datetime]
        # 创建唯一ID（时间+币种）
        df_submit["id"] = df_submit["datetime"].dt.strftime("%Y%m%d%H%M%S") + "_" + df_submit["symbol"]
        df_submit = df_submit[['id', 'predict_return']]

        # 处理提交ID文件
        if os.path.exists(self.submission_id_path):
            # 读取提交ID文件
            df_submission_id = pd.read_csv(self.submission_id_path)
            id_list = df_submission_id["id"].tolist()
            print(f"Submission ID count: {len(id_list)}")
            # 筛选在提交ID列表中的预测结果
            df_submit_competion = df_submit[df_submit['id'].isin(id_list)]
            # 找出缺失的ID
            missing_elements = list(set(id_list) - set(df_submit_competion['id']))
            print(f"Missing IDs: {len(missing_elements)}")
            # 为缺失ID创建默认预测值（0）
            new_rows = pd.DataFrame({'id': missing_elements, 'predict_return': [0] * len(missing_elements)})
            # 合并预测结果
            df_submit_competion = pd.concat([df_submit_competion, new_rows], ignore_index=True)
        else:
            print(f"Warning: {self.submission_id_path} not found. Saving submission without ID filtering.")
            df_submit_competion = df_submit

        # 保存提交文件
        print("Submission file sample:", df_submit_competion.head())
        df_submit_competion.to_csv("submit.csv", index=False)

        # 准备真实值检查文件
        df_check = data.reset_index(level=0)
        df_check = df_check[['level_0', 'target']]
        df_check['symbol'] = df_check.index.values
        df_check = df_check[['level_0', 'symbol', 'target']]
        df_check.columns = ['datetime', 'symbol', 'true_return']
        df_check = df_check[df_check['datetime'] >= self.start_datetime]
        df_check["id"] = df_check["datetime"].dt.strftime("%Y%m%d%H%M%S") + "_" + df_check["symbol"]
        df_check = df_check[['id', 'true_return']]
        df_check.to_csv("check.csv", index=False)

        # 计算整体加权Spearman相关系数
        rho_overall = self.weighted_spearmanr(data['target'], data['y_pred'])
        print(f"Weighted Spearman correlation coefficient: {rho_overall:.4f}")

        # SHAP分析（解释模型预测）
        explainer = shap.Explainer(best_model)  # 创建解释器
        shap_values = explainer(X_scaled)  # 计算SHAP值
        shap.summary_plot(shap_values, X.columns)  # 绘制SHAP摘要图

    def run(self):
        # 主运行函数
        # 加载所有加密货币数据
        all_symbol_list, time_arr, vwap_arr, amount_arr, atr_arr, macd_arr, buy_volume_arr, volume_arr = self.get_all_symbol_kline()
        if not all_symbol_list:
            print("No data loaded, exiting.")
            return

        print(f"all_symbol_list length: {len(all_symbol_list)}, vwap_arr shape: {vwap_arr.shape}")

        # 创建各种指标的DataFrame
        df_vwap = pd.DataFrame(vwap_arr, columns=all_symbol_list, index=time_arr)  # VWAP
        df_amount = pd.DataFrame(amount_arr, columns=all_symbol_list, index=time_arr)  # 交易额
        df_atr = pd.DataFrame(atr_arr, columns=all_symbol_list, index=time_arr)  # ATR
        df_macd = pd.DataFrame(macd_arr, columns=all_symbol_list, index=time_arr)  # MACD
        df_buy_volume = pd.DataFrame(buy_volume_arr, columns=all_symbol_list, index=time_arr)  # 买入量
        df_volume = pd.DataFrame(volume_arr, columns=all_symbol_list, index=time_arr)  # 总交易量

        # 定义时间窗口
        windows_1d = 4 * 24 * 1  # 1天（15分钟数据）
        windows_7d = 4 * 24 * 7  # 7天
        windows_4h = 4 * 4  # 4小时

        # 计算动量因子
        df_4h_momentum = (df_vwap / df_vwap.shift(windows_4h) - 1).replace([np.inf, -np.inf], np.nan).fillna(0)  # 4小时动量
        df_7d_momentum = (df_vwap / df_vwap.shift(windows_7d) - 1).replace([np.inf, -np.inf], np.nan).fillna(0)  # 7天动量

        # 计算交易量因子
        df_amount_sum = df_amount.rolling(windows_7d).sum().replace([np.inf, -np.inf], np.nan).fillna(0)  # 7天总交易额
        df_vol_momentum = (df_amount / df_amount.shift(windows_1d) - 1).replace([np.inf, -np.inf], np.nan).fillna(
            0)  # 交易额动量

        # 计算市场压力因子
        df_buy_pressure = (df_buy_volume - (df_volume - df_buy_volume)).replace([np.inf, -np.inf], np.nan).fillna(
            0)  # 买入压力

        # 计算24小时收益率（目标变量）
        df_24hour_rtn = (df_vwap / df_vwap.shift(windows_1d) - 1).replace([np.inf, -np.inf], np.nan).fillna(0)

        # 训练模型（使用滞后的24小时收益率作为目标）
        self.train(df_24hour_rtn.shift(-windows_1d), df_4h_momentum, df_7d_momentum, df_amount_sum, df_vol_momentum,
                   df_atr,
                   df_macd, df_buy_pressure)


if __name__ == '__main__':
    # 程序入口
    model = OptimizedModel()  # 创建模型实例
    model.run()  # 运行模型