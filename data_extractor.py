import requests
import json
import os
import csv
import logging
import random
from typing import List, Dict
from requests.auth import HTTPBasicAuth
# 配置日志记录器
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('data_extractor.log')
    ]
)
logger = logging.getLogger(__name__)

class DataExtractor:
    """用于从WorldQuant Brain获取数据字段和运算符并保存到CSV文件的工具类"""
    def __init__(self, credentials_path: str = './credential.txt'):
        """初始化数据提取器
        
        参数:
            credentials_path: 包含API认证信息的文件路径
        """
        self.sess = requests.Session()
        self.credentials_path = credentials_path
        self.setup_auth(credentials_path)
        
    def setup_auth(self, credentials_path: str) -> None:
        """设置与WorldQuant Brain的认证
        
        参数:
            credentials_path: 包含API认证信息的文件路径
        """
        logging.info(f"从 {credentials_path} 加载凭据")
        with open(credentials_path) as f:
            credentials = json.load(f)
        
        username, password = credentials
        self.sess.auth = HTTPBasicAuth(username, password)
        
        logging.info("正在与WorldQuant Brain进行认证...")
        response = self.sess.post('https://api.worldquantbrain.com/authentication')
        logging.info(f"认证响应状态: {response.status_code}")
        
        if response.status_code != 201:
            raise Exception(f"认证失败: {response.text}")
    
    def get_data_fields(self, search: str = '', use_specific_datasets: bool = True) -> List[Dict]:
        """从WorldQuant Brain获取可用的数据字段
        
        参数:
            search: 搜索关键词，用于筛选数据字段
            use_specific_datasets: 是否使用指定的数据集列表，True时使用预设的数据集列表
            
        返回:
            List[Dict]: 包含数据字段信息的字典列表
        """
        import pandas as pd
        
        # 定义搜索范围
        searchScope = {
            'instrumentType': 'EQUITY',
            'region': 'USA',
            'delay': 1,
            'universe': 'TOP3000'
        }
        
        try:
            all_datafields_df = pd.DataFrame()
            
            if use_specific_datasets:
                # 要查询的数据集列表，与alpha-generator.py保持一致
                #'fundamental6', 'fundamental2', 'analyst4', 'model16', 'model51', 'news12'
                datasets = ['fundamental6']
                
                for dataset_id in datasets:
                    print(f"正在获取数据集 {dataset_id} 的字段...")
                    dataset_df = self._get_datafields_impl(self.sess, searchScope, dataset_id, search)
                    # 添加数据集标识
                    if not dataset_df.empty:
                        dataset_df['dataset_id'] = dataset_id
                        all_datafields_df = pd.concat([all_datafields_df, dataset_df], ignore_index=True)
                
                print(f"所有指定数据集的字段总数: {len(all_datafields_df)}")
            else:
                # 获取所有数据集
                all_datafields_df = self._get_datafields_impl(self.sess, searchScope, '', search)
                print(f"所有数据集的字段总数: {len(all_datafields_df)}")
            
            # 转换DataFrame为字典列表
            datafields_list = all_datafields_df.to_dict('records')
            print(f"找到的唯一字段总数: {len(datafields_list)}")
            return datafields_list
        except Exception as e:
            logger.error(f"获取数据字段失败: {e}")
            return []
    
    def _get_datafields_impl(self, s, searchScope, dataset_id: str = '', search: str = ''):
        """内部实现方法，直接使用用户提供的代码"""
        import pandas as pd
        instrument_type = searchScope['instrumentType']
        region = searchScope['region']
        delay = searchScope['delay']
        universe = searchScope['universe']
    
        if len(search) == 0:
            url_template = "https://api.worldquantbrain.com/data-fields?" + \
                           f"instrumentType={instrument_type}" + \
                           f"&region={region}&delay={str(delay)}&universe={universe}&dataset.id={dataset_id}&limit=50" + \
                           "&offset={x}"
            count = s.get(url_template.format(x=0)).json()['count']
        else:
            url_template = "https://api.worldquantbrain.com/data-fields?" + \
                           f"instrumentType={instrument_type}" + \
                           f"&region={region}&delay={str(delay)}&universe={universe}&limit=50" + \
                           f"&search={search}" + \
                           "&offset={x}"
            count = 100
    
        datafields_list = []
        for x in range(0, count, 50):
            datafields = s.get(url_template.format(x=x))
            datafields_list.append(datafields.json()['results'])
    
        datafields_list_flat = [item for sublist in datafields_list for item in sublist]
    
        datafields_df = pd.DataFrame(datafields_list_flat)
        return datafields_df

    def get_operators(self) -> List[Dict]:
        """从WorldQuant Brain获取可用的运算符
        
        返回:
            List[Dict]: 包含运算符信息的字典列表
        """
        print("正在请求运算符...")
        response = self.sess.get('https://api.worldquantbrain.com/operators')
        print(f"运算符响应状态: {response.status_code}")
        print(f"运算符响应: {response.text[:500]}...")  # 打印前500个字符
        
        if response.status_code != 200:
            raise Exception(f"获取运算符失败: {response.text}")
        
        data = response.json()
        # 运算符端点可能直接返回数组，而不是带有'items'或'results'的对象
        if isinstance(data, list):
            return data
        elif 'results' in data:
            return data['results']
        else:
            raise Exception(f"意外的运算符响应格式。响应: {data}")

    def save_data_fields_to_csv(self, data_fields: List[Dict], filename: str = 'data_fields.csv'):
        """将数据字段保存到CSV文件
        
        参数:
            data_fields: 从API获取的数据字段列表
            filename: 要保存的CSV文件名
        """
        if not data_fields:
            print("没有数据字段可供保存")
            return
        
        # 确保输出目录存在
        os.makedirs(os.path.dirname(os.path.abspath(filename)) or '.', exist_ok=True)
        
        # 确定CSV文件的字段名
        fieldnames = ['id', 'name', 'description', 'category', 'dataset', 'type', 'unit', 'frequency']
        
        print(f"正在将 {len(data_fields)} 个数据字段保存到 {filename}...")
        with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for field in data_fields:
                # 确保所有字段都有值
                row = {
                    'id': field.get('id', ''),
                    'name': field.get('name', ''),
                    'description': field.get('description', ''),
                    'category': field.get('category', ''),
                    'dataset': field.get('dataset', {}).get('name', '') if isinstance(field.get('dataset'), dict) else '',
                    'type': field.get('type', ''),
                    'unit': field.get('unit', ''),
                    'frequency': field.get('frequency', '')
                }
                writer.writerow(row)
        
        print(f"数据字段已成功保存到 {filename}")

    def save_operators_to_csv(self, operators: List[Dict], filename: str = 'operators.csv'):
        """将运算符保存到CSV文件
        
        参数:
            operators: 从API获取的运算符列表
            filename: 要保存的CSV文件名
        """
        if not operators:
            print("没有运算符可供保存")
            return
        
        # 确保输出目录存在
        os.makedirs(os.path.dirname(os.path.abspath(filename)) or '.', exist_ok=True)
        
        # 确定CSV文件的字段名
        fieldnames = ['id', 'name', 'category', 'type', 'definition', 'description']
        
        print(f"正在将 {len(operators)} 个运算符保存到 {filename}...")
        with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for op in operators:
                # 确保所有字段都有值
                row = {
                    'id': op.get('id', ''),
                    'name': op.get('name', ''),
                    'category': op.get('category', ''),
                    'type': op.get('type', 'SCALAR'),
                    'definition': op.get('definition', ''),
                    'description': op.get('description', '')
                }
                writer.writerow(row)
        
        print(f"运算符已成功保存到 {filename}")

    def extract_and_save_all(self, fields_filename: str = 'data_fields.csv', operators_filename: str = 'operators.csv', use_specific_datasets: bool = True):
        """提取所有数据字段和运算符并保存到CSV文件
        
        参数:
            fields_filename: 数据字段CSV文件名
            operators_filename: 运算符CSV文件名
            use_specific_datasets: 是否使用指定的数据集列表
        """
        try:
            # 获取数据字段并保存
            data_fields = self.get_data_fields(use_specific_datasets=use_specific_datasets)
            if data_fields:
                self.save_data_fields_to_csv(data_fields, fields_filename)
            
            # 获取运算符并保存
            operators = self.get_operators()
            if operators:
                self.save_operators_to_csv(operators, operators_filename)
            
            print("所有数据提取和保存操作已完成！")
        except Exception as e:
            logger.error(f"提取和保存数据时发生错误: {e}")
            raise

if __name__ == "__main__":
    """主函数，演示如何使用DataExtractor类"""
    try:
        # 创建数据提取器实例
        extractor = DataExtractor()
        
        # 提取并保存所有数据
        extractor.extract_and_save_all()
        
    except Exception as e:
        print(f"程序执行失败: {e}")