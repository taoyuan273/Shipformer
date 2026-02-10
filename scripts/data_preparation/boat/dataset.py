import json
import os
import numpy as np
import pandas as pd
from pathlib import Path


dataset_name = 'Boat'
data_file_path = r'week_2292_c25_p6_selected.csv'
current_file = Path(__file__).resolve()
project_root = current_file.parents[3]
output_dir = project_root / "datasets" / dataset_name
output_dir.mkdir(parents=True, exist_ok=True)
graph_file_path = None  # 如果没有图结构文件，可以保持 None
target_channel = [0]

# 时间特征
add_time_of_day = True
add_day_of_week = True
add_day_of_month = False
add_day_of_year = False

# 一天86400 秒
steps_per_day = 86400
frequency = 0.01/60

# 特征描述
domain = 'my custom domain: second-based in a week'
feature_description = [domain, 'time_of_day', 'day_of_week']

# 常规训练设置
regular_settings = {
    'INPUT_LEN': 3000,
    'OUTPUT_LEN': 1000,
    'TRAIN_VAL_TEST_RATIO': [0.8, 0.1, 0.1],
    'NORM_EACH_CHANNEL': True,
    'RESCALE': False,
    'METRICS': ['MAE', 'MSE'],
    'NULL_VAL': np.nan
}


def load_and_preprocess_data():
    """
    读取并预处理数据:
    """
    df = pd.read_csv(data_file_path)
    week_sec = df['周内秒'].values
    numeric_cols = [col for col in df.columns if col != 'week_second']
    sub_df = df[numeric_cols]
    # 扩展到三维 [L, N, 1]
    data = np.expand_dims(sub_df.values, axis=-1)
    # 选择目标通道
    data = data[..., target_channel]
    print(f'Raw time series shape: {data.shape}')
    return data, df


def add_temporal_features(data, df):
    """
    时间特征计算
    """
    l, n, _ = data.shape
    feature_list = [data]

    week_sec = df['周内秒'].values

    if add_time_of_day:
        time_of_day = (week_sec % 86400) / 86400
        time_of_day_tiled = np.tile(time_of_day, (1, n, 1)).transpose((2, 1, 0))
        feature_list.append(time_of_day_tiled)

    if add_day_of_week:
        day_of_week = (week_sec // 86400) / 7
        day_of_week_tiled = np.tile(day_of_week, (1, n, 1)).transpose((2, 1, 0))
        feature_list.append(day_of_week_tiled)

    if add_day_of_month:
        pass

    if add_day_of_year:
        pass

    data_with_features = np.concatenate(feature_list, axis=-1)
    return data_with_features


def save_data(data):
    """
    保存数据到二进制输出
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    file_path = os.path.join(output_dir, 'data.dat')
    fp = np.memmap(file_path, dtype='float32', mode='w+', shape=data.shape)
    fp[:] = data[:]
    fp.flush()
    del fp
    print(f'Data saved to {file_path}')


def save_description(data):
    """
    将数据集描述信息写入json。
    """
    description = {
        'name': dataset_name,
        'domain': domain,
        'shape': data.shape,
        'num_time_steps': data.shape[0],
        'num_nodes': data.shape[1],
        'num_features': data.shape[2],
        'feature_description': feature_description,
        'has_graph': graph_file_path is not None,
        'frequency (minutes)': frequency,
        'regular_settings': regular_settings
    }
    description_path = os.path.join(output_dir, 'desc.json')
    with open(description_path, 'w') as f:
        json.dump(description, f, indent=4)
    print(f'Description saved to {description_path}')
    print(description)


def main():
    data, df = load_and_preprocess_data()
    data_with_features = add_temporal_features(data, df)
    save_data(data_with_features)
    save_description(data_with_features)


if __name__ == '__main__':
    main()
