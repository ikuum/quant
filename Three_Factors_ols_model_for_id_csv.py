import numpy as np
import pandas as pd
import datetime, os, time
import multiprocessing as mp
from sklearn.linear_model import LinearRegression


class OlsModel:
    # 初始化，该类包含训练数据路径和开始日期两个实体
    def __init__(self):
        # 设置序列数据的文件夹路径
        self.train_data_path = "train_data"
        self.start_datetime = datetime.datetime(2021, 3, 1, 0, 0, 0)

    # 得到所有币种的符号（用来读取对应币种的数据文件）
    def get_all_symbol_list(self):
        # 获取训练数据目录中的所有文件名
        parquet_name_list = os.listdir(self.train_data_path)
        # 移除文件扩展名，只保留货币代码符号以生成货币代码列表
        symbol_list = [parquet_name.split(".")[0] for parquet_name in parquet_name_list]
        return symbol_list

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
        except Exception as e:
            print(f"get_single_symbol_kline_data error: {e}")
            df = pd.DataFrame()
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
        # 创建一个进程池，使用可用CPU核心数减2进行并行处理
        pool = mp.Pool(mp.cpu_count() - 2)
        # 获取所有货币的列表
        all_symbol_list = self.get_all_symbol_list()
        # 初始化列表用于存储每个异步读取任务返回的结果
        df_list = []
        for i in range(len(all_symbol_list)):
            df_list.append(pool.apply_async(self.get_single_symbol_kline_data, (all_symbol_list[i],)))
        # 关闭进程池，不再接受新任务
        pool.close()
        # 等待所有异步任务完成
        pool.join()
        # 收集所有异步结果的开盘价序列并按列连接成DataFrame，然后按时间升序排序
        df_open_price = pd.concat([i.get()['open_price'] for i in df_list], axis=1).sort_index(ascending=True)
        # 将时间索引（毫秒）转换为datetime类型数组
        time_arr = pd.to_datetime(pd.Series(df_open_price.index), unit="ms").values
        # 从DataFrame中获取开盘价的值并转换为float类型的NumPy数组
        open_price_arr = df_open_price.values.astype(float)
        # 从DataFrame中获取最高价的值并转换为float类型的NumPy数组
        high_price_arr = pd.concat([i.get()['high_price'] for i in df_list], axis=1).sort_index(ascending=True).values
        # 从DataFrame中获取最低价的值并转换为float类型的NumPy数组
        low_price_arr = pd.concat([i.get()['low_price'] for i in df_list], axis=1).sort_index(ascending=True).values
        # 从DataFrame中获取收盘价的值并转换为float类型的NumPy数组
        close_price_arr = pd.concat([i.get()['close_price'] for i in df_list], axis=1).sort_index(ascending=True).values
        # 收集所有货币的成交量加权平均价格序列并按列连接成数组
        vwap_arr = pd.concat([i.get()['vwap'] for i in df_list], axis=1).sort_index(ascending=True).values
        # 收集所有货币的交易量序列并按列连接成数组
        amount_arr = pd.concat([i.get()['amount'] for i in df_list], axis=1).sort_index(ascending=True).values
        print(f"完成获取所有币种k线数据, 耗时 {datetime.datetime.now() - t0}")
        return all_symbol_list, time_arr, open_price_arr, high_price_arr, low_price_arr, close_price_arr, vwap_arr, amount_arr



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

    def train(self, df_target, df_factor1, df_factor2, df_factor3):
        # 将因子1数据转换为长格式（多索引：时间和符号）
        factor1_long = df_factor1.stack()
        # 将因子2数据转换为长格式
        factor2_long = df_factor2.stack()
        # 将因子3数据转换为长格式
        factor3_long = df_factor3.stack()
        # 将目标数据转换为长格式
        target_long = df_target.stack()
        # 设置因子1系列的名称为'factor1'
        factor1_long.name = 'factor1'
        # 设置因子2系列的名称为'factor2'
        factor2_long.name = 'factor2'
        # 设置因子3系列的名称为'factor3'
        factor3_long.name = 'factor3'
        # 设置目标系列的名称为'target'
        target_long.name = 'target'
        # 将四个系列（因子1、因子2、因子3、目标）按列合并为单个DataFrame
        data = pd.concat([factor1_long, factor2_long, factor3_long, target_long], axis=1)
        # 删除包含缺失值（NaN）的行
        data = data.dropna()
        # 构建特征矩阵X，包含三个因子列
        X = data[['factor1', 'factor2', 'factor3']]
        # 构建目标变量y，即目标值列
        y = data['target']
        # 使用LinearRegression（多元线性回归）拟合模型
        model = LinearRegression()
        model.fit(X, y)

        # 输出回归模型的系数和截距
        print("线性回归模型系数:", model.coef_)
        print("线性回归模型截距:", model.intercept_)

        # 在原始数据中添加一列存储模型对每个样本的预测值
        data['y_pred'] = model.predict(X)
        df_submit = data.reset_index(level=0)
        df_submit = df_submit[['level_0', 'y_pred']]
        df_submit['symbol'] = df_submit.index.values
        df_submit = df_submit[['level_0', 'symbol', 'y_pred']]
        df_submit.columns = ['datetime', 'symbol', 'predict_return']
        df_submit = df_submit[df_submit['datetime'] >= self.start_datetime]
        df_submit["id"] = df_submit["datetime"].astype(str) + "_" + df_submit["symbol"]
        df_submit = df_submit[['id', 'predict_return']]

        print(df_submit)

        df_submission_id = pd.read_csv("submission_id.csv")
        id_list = df_submission_id["id"].tolist()
        df_submit_competion = df_submit[df_submit['id'].isin(id_list)]
        missing_elements = list(set(id_list) - set(df_submit_competion['id']))
        new_rows = pd.DataFrame({'id': missing_elements, 'predict_return': [0] * len(missing_elements)})
        df_submit_competion = pd.concat([df_submit, new_rows], ignore_index=True)
        print(df_submit_competion.shape)
        df_submit_competion.to_csv("submit.csv", index=False)

        df_check = data.reset_index(level=0)
        df_check = df_check[['level_0', 'target']]
        df_check['symbol'] = df_check.index.values
        df_check = df_check[['level_0', 'symbol', 'target']]
        df_check.columns = ['datetime', 'symbol', 'true_return']
        df_check = df_check[df_check['datetime'] >= self.start_datetime]
        df_check["id"] = df_check["datetime"].astype(str) + "_" + df_check["symbol"]
        df_check = df_check[['id', 'true_return']]

        print(df_check)

        df_check.to_csv("check.csv", index=False)

        # 在整个样本上计算加权Spearman相关系数
        rho_overall = self.weighted_spearmanr(data['target'], data['y_pred'])
        print(f"加权Spearman相关系数: {rho_overall:.4f}")

    def run(self):
        # 调用get_all_symbol_kline函数获取所有货币的K线数据和事件数据
        all_symbol_list, time_arr, open_price_arr, high_price_arr, low_price_arr, close_price_arr, vwap_arr, amount_arr = self.get_all_symbol_kline()
        # 将vwap数组转换为DataFrame，货币代码作为列，时间作为索引（下一行设置索引）
        df_vwap = pd.DataFrame(vwap_arr, columns=all_symbol_list, index=time_arr)
        # 将amount数组转换为DataFrame，货币代码作为列，时间作为索引
        df_amount = pd.DataFrame(amount_arr, columns=all_symbol_list, index=time_arr)
        windows_1d = 4 * 24 * 1  # 1天对应的窗口大小（15分钟K线）
        windows_7d = 4 * 24 * 7  # 7天对应的窗口大小（15分钟K线）
        # 使用滚动计算计算过去24小时的收益率
        df_24hour_rtn = df_vwap / df_vwap.shift(windows_1d) - 1
        # 使用滚动计算计算过去15分钟的收益率
        df_15min_rtn = df_vwap / df_vwap.shift(1) - 1
        # 计算第一个因子：7天波动率因子，反映市场风险和不确定性
        df_7d_volatility = df_15min_rtn.rolling(windows_7d).std(ddof=1)
        # 计算第二个因子：7天动量因子，正动量表示上涨趋势，负动量表示下跌趋势
        df_7d_momentum = df_vwap / df_vwap.shift(windows_7d) - 1
        # 计算第三个因子：7天总交易量因子，高交易量通常表示市场关注度高
        df_amount_sum = df_amount.rolling(windows_7d).sum()
        # 调用train方法，使用滞后的7天24小时收益率作为目标值，三个因子作为输入进行模型训练
        self.train(df_24hour_rtn.shift(-windows_1d), df_7d_volatility, df_7d_momentum, df_amount_sum)


if __name__ == '__main__':
    model = OlsModel()
    model.run()