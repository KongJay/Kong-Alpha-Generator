import argparse
import requests
import json
import os
from time import sleep
from requests.auth import HTTPBasicAuth
from typing import List, Dict
import time
import re
import logging
from queue import Queue
from threading import Thread
from concurrent.futures import ThreadPoolExecutor, as_completed
import random

# 配置日志记录器
logger = logging.getLogger(__name__)

class RetryQueue:
    """重试队列类，用于管理需要重新测试的Alpha表达式
    
    该类维护一个队列，存储由于模拟限制等原因需要重新测试的Alpha表达式。
    它会在后台线程中自动处理队列中的任务，根据最大重试次数进行重试。
    """
    def __init__(self, generator, max_retries=3, retry_delay=60):
        """初始化重试队列
        
        参数:
            generator: AlphaGenerator实例，用于调用测试方法
            max_retries: 最大重试次数，默认为3次
            retry_delay: 重试间隔时间（秒），默认为60秒
        """
        self.queue = Queue()  # 存储待重试的Alpha表达式队列
        self.max_retries = max_retries  # 最大重试次数
        self.retry_delay = retry_delay  # 重试延迟时间（秒）
        self.generator = generator  # 存储对生成器的引用，用于调用测试方法
        self.worker = Thread(target=self._process_queue, daemon=True)  # 后台工作线程
        self.worker.start()  # 启动工作线程
    
    def add(self, alpha: str, retry_count: int = 0):
        """向队列中添加一个需要重试的Alpha表达式
        
        参数:
            alpha: Alpha表达式字符串
            retry_count: 当前重试次数，默认为0
        """
        self.queue.put((alpha, retry_count))  # 将Alpha表达式和重试次数放入队列
    
    def _process_queue(self):
        """处理队列中的任务，在后台线程中运行
        
        不断检查队列，对队列中的Alpha表达式进行重试测试，直到达到最大重试次数。
        如果模拟限制超出，将表达式重新加入队列；否则将结果添加到生成器的结果列表中。
        """
        while True:
            if not self.queue.empty():
                alpha, retry_count = self.queue.get()  # 从队列中获取Alpha表达式和重试次数
                if retry_count >= self.max_retries:
                    logging.error(f"Alpha表达式超过最大重试次数: {alpha}")
                    continue
                    
                try:
                    # 调用生成器的_test_alpha_impl方法测试Alpha表达式，避免递归调用
                    result = self.generator._test_alpha_impl(alpha)
                    # 检查是否是模拟限制超出错误
                    if result.get("status") == "error" and "SIMULATION_LIMIT_EXCEEDED" in result.get("message", ""):
                        logging.info(f"模拟限制超出，重新排队Alpha表达式: {alpha}")
                        time.sleep(self.retry_delay)  # 等待重试延迟时间
                        self.add(alpha, retry_count + 1)  # 增加重试次数并重新加入队列
                    else:
                        # 将测试结果添加到生成器的结果列表中
                        self.generator.results.append({
                            "alpha": alpha,
                            "result": result
                        })
                except Exception as e:
                    logging.error(f"处理Alpha表达式时出错: {str(e)}")
                    
            time.sleep(1)  # 防止CPU空转

class AlphaGenerator:
    """Alpha生成器类，用于通过Ollama生成和测试Alpha因子表达式
    
    该类负责与WorldQuant Brain API和Ollama进行交互，生成Alpha因子表达式，
    提交到API进行测试，并处理测试结果。它支持批处理、并发测试、错误重试
    和模型降级等功能。
    """
    def __init__(self, credentials_path: str, ollama_url: str = "http://localhost:11434", max_concurrent: int = 2):
        """初始化Alpha生成器
        
        参数:
            credentials_path: 凭据文件路径，包含API认证信息
            ollama_url: Ollama API的URL地址，默认为"http://localhost:11434"
            max_concurrent: 最大并发测试数量，默认为2
        """
        self.sess = requests.Session()  # 创建HTTP会话
        self.credentials_path = credentials_path  # 存储凭据文件路径，用于重新认证
        self.setup_auth(credentials_path)  # 初始化认证
        self.ollama_url = ollama_url  # Ollama API的URL地址
        self.results = []  # 存储测试结果的列表
        self.pending_results = {}  # 存储待处理结果的字典
        self.retry_queue = RetryQueue(self)  # 创建重试队列实例
        
        # 为了防止VRAM问题，减少并发工作线程数量
        self.executor = ThreadPoolExecutor(max_workers=max_concurrent)  # 用于并发模拟的线程池
        self.vram_cleanup_interval = 10  # 每10次操作进行一次VRAM清理
        self.operation_count = 0  # 操作计数器
        
        # 模型降级跟踪相关变量
        self.initial_model = getattr(self, 'model_name', 'deepseek-r1:8b')  # 初始模型
        self.error_count = 0  # 错误计数器
        self.max_errors_before_downgrade = 3  # 降级模型前的最大错误数
        # 模型列表，按资源消耗从高到低排序，用于降级
        self.model_fleet = [
            'deepseek-r1:8b',   # 主要模型
            'deepseek-r1:7b',   # 第一个备用模型
            'deepseek-r1:1.5b', # 第二个备用模型
            'llama3:3b',        # 第三个备用模型
            'phi3:mini'         # 紧急备用模型
        ]
        self.current_model_index = 0  # 当前使用的模型索引（在model_fleet中的位置）
        
    def setup_auth(self, credentials_path: str) -> None:
        """设置与WorldQuant Brain的认证
        
        参数:
            credentials_path: 包含API认证信息的文件路径
            
        异常:
            Exception: 当认证失败时抛出异常
        """
        logging.info(f"从 {credentials_path} 加载凭据")
        with open(credentials_path) as f:
            credentials = json.load(f)  # 加载凭据文件
        
        username, password = credentials  # 提取用户名和密码
        self.sess.auth = HTTPBasicAuth(username, password)  # 设置HTTP基本认证
        
        logging.info("正在与WorldQuant Brain进行认证...")
        response = self.sess.post('https://api.worldquantbrain.com/authentication')
        logging.info(f"认证响应状态: {response.status_code}")
        logging.debug(f"认证响应: {response.text[:500]}...")  # 只打印前500个字符
        
        if response.status_code != 201:
            raise Exception(f"认证失败: {response.text}")
    
    def cleanup_vram(self):
        """执行VRAM清理，通过强制垃圾回收和等待来释放GPU内存
        
        该方法尝试导入gc模块并执行垃圾回收，以释放不再使用的GPU内存。
        清理后会等待2秒钟，以便GPU内存有时间被完全释放。
        """
        try:
            import gc
            gc.collect()  # 执行垃圾回收
            logging.info("已执行VRAM清理")
            # 添加短暂延迟以允许GPU内存被释放
            time.sleep(2)
        except Exception as e:
            logging.warning(f"VRAM清理失败: {e}")
        
    def get_data_fields(self) -> List[Dict]:
        """从WorldQuant Brain获取多个数据集中的可用数据字段，并进行随机采样
        
        返回:
            List[Dict]: 包含数据字段信息的字典列表
        """
        # 要查询的数据集列表
        datasets = ['fundamental6', 'fundamental2', 'analyst4', 'model16', 'model51', 'news12']
        all_fields = []  # 存储所有获取到的数据字段
        
        # 基础查询参数
        base_params = {
            'delay': 1,  # 延迟
            'instrumentType': 'EQUITY',  # 工具类型为股票
            'limit': 20,  # 每页限制20个结果
            'region': 'USA',  # 地区为美国
            'universe': 'TOP3000'  # 股票池为TOP3000
        }
        
        try:
            print("正在从多个数据集请求数据字段...")
            for dataset in datasets:
                # 先获取该数据集的字段总数
                params = base_params.copy()
                params['dataset.id'] = dataset  # 设置数据集ID
                params['limit'] = 1  # 只获取一条记录以高效地获取总数
                
                print(f"正在获取数据集 {dataset} 的字段数量")
                count_response = self.sess.get('https://api.worldquantbrain.com/data-fields', params=params)
                
                if count_response.status_code == 200:
                    count_data = count_response.json()
                    total_fields = count_data.get('count', 0)  # 获取字段总数
                    print(f"数据集 {dataset} 中的字段总数: {total_fields}")
                    
                    if total_fields > 0:
                        # 生成随机偏移量以获取随机采样
                        max_offset = max(0, total_fields - base_params['limit'])
                        random_offset = random.randint(0, max_offset)
                        
                        # 获取随机子集
                        params['offset'] = random_offset
                        params['limit'] = min(20, total_fields)  # 不要超过总字段数
                        
                        print(f"正在获取数据集 {dataset} 的字段，偏移量为 {random_offset}")
                        response = self.sess.get('https://api.worldquantbrain.com/data-fields', params=params)
                        
                        if response.status_code == 200:
                            data = response.json()
                            fields = data.get('results', [])
                            print(f"在数据集 {dataset} 中找到 {len(fields)} 个字段")
                            all_fields.extend(fields)  # 将找到的字段添加到列表中
                        else:
                            print(f"获取数据集 {dataset} 的字段失败: {response.text[:500]}")
                else:
                    print(f"获取数据集 {dataset} 的数量失败: {count_response.text[:500]}")
            
            # 移除重复字段
            unique_fields = {field['id']: field for field in all_fields}.values()
            print(f"找到的唯一字段总数: {len(unique_fields)}")
            return list(unique_fields)
            
        except Exception as e:
            logger.error(f"获取数据字段失败: {e}")
            return []

    def get_operators(self) -> List[Dict]:
        """从WorldQuant Brain获取可用的运算符
        
        返回:
            List[Dict]: 包含运算符信息的字典列表
            
        异常:
            Exception: 当获取运算符失败或响应格式不符合预期时抛出异常
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

    def clean_alpha_ideas(self, ideas: List[str]) -> List[str]:
        """清理和验证alpha表达式想法，只保留有效的表达式
        
        参数:
            ideas: Alpha表达式想法列表
            
        返回:
            List[str]: 清理后的有效Alpha表达式列表
        """
        cleaned_ideas = []  # 存储清理后的有效表达式
        
        for idea in ideas:
            # 如果想法只是数字或单个单词，则跳过
            if re.match(r'^\d+\.?$|^[a-zA-Z]+$', idea):
                continue
            
            # 如果想法是描述性文本（包含常见英文单词），则跳过
            common_words = ['it', 'the', 'is', 'are', 'captures', 'provides', 'measures']
            if any(word in idea.lower() for word in common_words):
                continue
            
            # 验证想法是否包含有效的运算符/函数
            valid_functions = ['ts_mean', 'divide', 'subtract', 'add', 'multiply', 'zscore', 
                              'ts_rank', 'ts_std_dev', 'rank', 'log', 'sqrt']
            if not any(func in idea for func in valid_functions):
                continue
            
            cleaned_ideas.append(idea)  # 直接添加有效的表达式字符串
        
        return cleaned_ideas

    def generate_alpha_ideas_with_ollama(self, data_fields: List[Dict], operators: List[Dict]) -> List[str]:
        """使用Ollama与FinGPT模型生成alpha因子表达式想法
        
        参数:
            data_fields: 从WorldQuant Brain获取的数据字段列表
            operators: 从WorldQuant Brain获取的运算符列表
            
        返回:
            List[str]: 生成的alpha因子表达式列表
        """
        print("正在按类别组织运算符...")
        operator_by_category = {}  # 按类别存储运算符的字典
        for op in operators:
            category = op['category']
            if category not in operator_by_category:
                operator_by_category[category] = []
            operator_by_category[category].append({
                'name': op['name'],
                'type': op.get('type', 'SCALAR'),  # 如果没有指定类型，默认为SCALAR
                'definition': op['definition'],
                'description': op['description']
            })

        try:
            # 如果之前尝试时遇到了token限制，清除已测试的表达式
            if hasattr(self, '_hit_token_limit'):
                logging.info("由于之前的token限制，正在清除已测试的表达式")
                self.results = []
                delattr(self, '_hit_token_limit')

            # 从每个类别中随机采样约50%的运算符
            sampled_operators = {}
            for category, ops in operator_by_category.items():
                sample_size = max(1, int(len(ops) * 0.5))  # 每个类别至少1个运算符
                sampled_operators[category] = random.sample(ops, sample_size)

            print("正在为FinGPT准备提示词...")
            # 格式化运算符，包含类型、定义和描述
            def format_operators(ops):
                formatted = []
                for op in ops:
                    formatted.append(f"{op['name']} ({op['type']})\n"
                                   f"  Definition: {op['definition']}\n"
                                   f"  Description: {op['description']}")
                return formatted

            prompt = f"""生成5个独特的alpha因子表达式，使用提供的运算符和数据字段。只返回表达式，每行一个，不要添加注释或解释。

可用数据字段:
{[field['id'] for field in data_fields]}

按类别分组的可用运算符:
时间序列:
{chr(10).join(format_operators(sampled_operators.get('Time Series', [])))}

横截面:
{chr(10).join(format_operators(sampled_operators.get('Cross Sectional', [])))}

算术运算:
{chr(10).join(format_operators(sampled_operators.get('Arithmetic', [])))}

逻辑运算:
{chr(10).join(format_operators(sampled_operators.get('Logical', [])))}

向量运算:
{chr(10).join(format_operators(sampled_operators.get('Vector', [])))}

转换运算:
{chr(10).join(format_operators(sampled_operators.get('Transformational', [])))}

分组运算:
{chr(10).join(format_operators(sampled_operators.get('Group', [])))}

要求:
1. 让你的直觉引导你。
2. 使用运算符和数据字段创建独特且可能有利可图的alpha因子。
3. 一切皆有可能 42。

提示: 
- 你可以使用分号来分隔多个表达式。
- 注意运算符类型（SCALAR, VECTOR, MATRIX）的兼容性。
- 研究运算符的定义和描述以理解其行为。

示例格式:
ts_std_dev(cashflow_op, 180)
rank(divide(revenue, assets))
market_ret = ts_product(1+group_mean(returns,1,market),250)-1;rfr = vec_avg(fnd6_newqeventv110_optrfrq);expected_return = rfr+beta_last_360_days_spy*(market_ret-rfr);actual_return = ts_product(returns+1,250)-1;actual_return-expected_return
"""

            # 准备Ollama API请求
            model_name = getattr(self, 'model_name', self.model_fleet[self.current_model_index])
            ollama_data = {
                'model': model_name,
                'prompt': prompt,
                'stream': False,  # 不使用流式响应
                'temperature': 0.3,  # 控制随机性，较低的值使输出更确定性
                'top_p': 0.9,  # 核采样参数
                'num_predict': 1000  # 预测的最大token数，Ollama使用num_predict而不是max_tokens
            }

            print("正在向Ollama API发送请求...")
            try:
                response = requests.post(
                    f'{self.ollama_url}/api/generate',
                    json=ollama_data,
                    timeout=360  # 6分钟超时
                )

                print(f"Ollama API响应状态: {response.status_code}")
                print(f"Ollama API响应: {response.text[:500]}...")  # 打印前500个字符

                if response.status_code == 500:
                    logging.error(f"Ollama API返回500错误: {response.text}")
                    # 遇到500错误时触发模型降级
                    self._handle_ollama_error("500_error")
                    return []
                elif response.status_code != 200:
                    raise Exception(f"Ollama API请求失败: {response.text}")
                    
            except requests.exceptions.Timeout:
                logging.error("Ollama API请求超时 (360s)")
                # 超时触发模型降级
                self._handle_ollama_error("timeout")
                return []
            except requests.exceptions.ConnectionError as e:
                if "Read timed out" in str(e):
                    logging.error("Ollama API读取超时")
                    # 读取超时触发模型降级
                    self._handle_ollama_error("read_timeout")
                    return []
                else:
                    raise e

            response_data = response.json()
            print(f"Ollama API响应JSON键: {list(response_data.keys())}")

            if 'response' not in response_data:
                raise Exception(f"意外的Ollama API响应格式: {response_data}")

            print("正在处理Ollama API响应...")
            content = response_data['response']
            
            # 提取纯alpha表达式的步骤:
            # 1. 移除markdown反引号
            # 2. 移除编号（例如 "1. ", "2. "）
            # 3. 跳过注释
            alpha_ideas = []
            for line in content.split('\n'):
                line = line.strip()
                if not line or line.startswith('#') or line.startswith('*'):
                    continue
                # 移除编号和反引号
                line = line.replace('`', '')
                if '. ' in line:
                    line = line.split('. ', 1)[1]  # 移除行首的编号
                if line and not line.startswith('Comment:'):
                    alpha_ideas.append(line)
            
            print(f"生成了 {len(alpha_ideas)} 个alpha想法")
            for i, alpha in enumerate(alpha_ideas, 1):
                print(f"Alpha {i}: {alpha}")
            
            # 清理和验证想法
            cleaned_ideas = self.clean_alpha_ideas(alpha_ideas)
            logging.info(f"找到 {len(cleaned_ideas)} 个有效的alpha表达式")
            
            return cleaned_ideas

        except Exception as e:
            if "token limit" in str(e).lower():
                self._hit_token_limit = True  # 标记已达到token限制
            logging.error(f"生成alpha想法时出错: {str(e)}")
            return []
    
    def _handle_ollama_error(self, error_type: str):
        """处理Ollama错误，必要时降级模型
        
        参数:
            error_type: 错误类型标识符（如"timeout", "500_error"等）
        """
        self.error_count += 1  # 增加错误计数
        logging.warning(f"Ollama错误 ({error_type}) - 计数: {self.error_count}/{self.max_errors_before_downgrade}")
        
        if self.error_count >= self.max_errors_before_downgrade:
            self._downgrade_model()  # 达到降级阈值时执行模型降级
            self.error_count = 0  # 降级后重置错误计数
    
    def _downgrade_model(self):
        """降级到模型列表中的下一个更小的模型
        
        当当前模型持续出现错误时，此方法会将模型切换到列表中的下一个更小的模型。
        如果已经是最小的模型，则重置回初始模型。
        """
        if self.current_model_index >= len(self.model_fleet) - 1:
            logging.error("已经在使用模型列表中最小的模型!")
            # 如果已经用尽所有选项，重置回初始模型
            self.current_model_index = 0
            self.model_name = self.initial_model
            logging.info(f"已重置到初始模型: {self.initial_model}")
            return
        
        old_model = self.model_fleet[self.current_model_index]
        self.current_model_index += 1  # 移动到列表中的下一个模型
        new_model = self.model_fleet[self.current_model_index]
        
        logging.warning(f"正在降级模型: {old_model} -> {new_model}")
        self.model_name = new_model
        
        # 如果存在编排器，则更新其中的模型
        try:
            # 尝试更新编排器的模型列表管理器
            if hasattr(self, 'orchestrator') and hasattr(self.orchestrator, 'model_fleet_manager'):
                self.orchestrator.model_fleet_manager.current_model_index = self.current_model_index
                self.orchestrator.model_fleet_manager.save_state()
                logging.info(f"已更新编排器模型列表为: {new_model}")
        except Exception as e:
            logging.warning(f"无法更新编排器模型列表: {e}")
        
        logging.info(f"已成功降级到 {new_model}")

    def test_alpha_batch(self, alphas: List[str]) -> None:
        """批量提交alpha表达式进行测试，包含监控功能，遵守并发限制
        
        参数:
            alphas: 要测试的alpha表达式列表
        """
        logging.info(f"开始批量测试 {len(alphas)} 个alpha")
        for alpha in alphas:
            logging.info(f"Alpha表达式: {alpha}")
        
        # 以较小的批次提交alpha，以遵守并发限制
        max_concurrent = self.executor._max_workers  # 最大并发数
        submitted = 0  # 已提交的数量
        queued = 0
        
        for i in range(0, len(alphas), max_concurrent):
            chunk = alphas[i:i + max_concurrent]
            logging.info(f"正在提交批次 {i//max_concurrent + 1}/{(len(alphas)-1)//max_concurrent + 1}（{len(chunk)} 个alpha）")
            
            # 提交批次
            futures = []
            for j, alpha in enumerate(chunk, 1):
                logging.info(f"正在提交alpha {i+j}/{len(alphas)}")
                future = self.executor.submit(self._test_alpha_impl, alpha)  # 异步提交测试任务
                futures.append((alpha, future))
            
            # 处理该批次的结果
            for alpha, future in futures:
                try:
                    result = future.result()  # 获取异步任务的结果
                    if result.get("status") == "error":
                        if "SIMULATION_LIMIT_EXCEEDED" in result.get("message", ""):
                            self.retry_queue.add(alpha)  # 模拟限制超出时加入重试队列
                            queued += 1
                            logging.info(f"已加入重试队列: {alpha}")
                        else:
                            logging.error(f"Alpha {alpha} 的模拟错误: {result.get('message')}")
                        continue
                        
                    sim_id = result.get("result", {}).get("id")  # 获取模拟ID
                    progress_url = result.get("result", {}).get("progress_url")  # 获取进度查询URL
                    if sim_id and progress_url:
                        self.pending_results[sim_id] = {
                            "alpha": alpha,
                            "progress_url": progress_url,
                            "status": "pending",
                            "attempts": 0
                        }
                        submitted += 1
                        logging.info(f"已成功提交 {alpha} (ID: {sim_id})")
                        
                except Exception as e:
                    logging.error(f"提交alpha {alpha} 时出错: {str(e)}")
            
            # 批次之间等待以避免API过载
            if i + max_concurrent < len(alphas):
                logging.info(f"等待10秒后再处理下一批...")
                sleep(10)
        
        logging.info(f"批量提交完成: {submitted} 已提交, {queued} 已加入重试队列")
        
        # 监控进度，直到所有任务完成或需要重试
        total_successful = 0
        max_monitoring_time = 3600  # 最大监控时间为1小时
        start_time = time.time()
        
        while self.pending_results:
            # 检查是否超时
            if time.time() - start_time > max_monitoring_time:
                logging.warning(f"监控超时 ({max_monitoring_time}秒)，停止监控")
                logging.warning(f"剩余未完成的模拟: {list(self.pending_results.keys())}")
                break
                
            logging.info(f"正在监控 {len(self.pending_results)} 个未完成的模拟...")
            completed = self.check_pending_results()  # 检查未完成结果的状态
            total_successful += completed
            sleep(5)  # 检查之间等待5秒
        
        logging.info(f"批次完成: {total_successful} 个成功的模拟")
        return total_successful

    def check_pending_results(self) -> int:
        """检查所有未完成模拟的状态，包含适当的重试处理
        
        返回:
            int: 成功完成的模拟数量
        """
        completed = []  # 已完成的模拟ID列表
        retry_queue = []  # 需要重试的模拟ID列表
        successful = 0  # 成功完成的数量
        
        for sim_id, info in self.pending_results.items():
            if info["status"] == "pending":
                # 检查模拟是否等待时间过长（30分钟）
                if "start_time" not in info:
                    info["start_time"] = time.time()
                elif time.time() - info["start_time"] > 1800:  # 30分钟
                    logging.warning(f"模拟 {sim_id} 等待时间过长，标记为失败")
                    completed.append(sim_id)
                    continue
                try:
                    sim_progress_resp = self.sess.get(info["progress_url"])
                    logging.info(f"正在检查alpha的模拟 {sim_id}: {info['alpha'][:50]}...")
                    
                    # 处理速率限制
                    if sim_progress_resp.status_code == 429:
                        logging.info("达到速率限制，稍后重试")
                        continue
                        
                    # 处理模拟限制
                    if "SIMULATION_LIMIT_EXCEEDED" in sim_progress_resp.text:
                        logging.info(f"alpha的模拟限制已超出: {info['alpha']}")
                        retry_queue.append((info['alpha'], sim_id))
                        continue
                        
                    # 处理Retry-After响应头
                    retry_after = sim_progress_resp.headers.get("Retry-After")
                    if retry_after:
                        try:
                            wait_time = int(float(retry_after))  # 处理小数形式的重试时间，如"2.5"
                            logging.info(f"下次检查前需要等待 {wait_time}秒")
                            time.sleep(wait_time)
                        except (ValueError, TypeError):
                            logging.warning(f"无效的Retry-After响应头: {retry_after}，使用默认的5秒")
                            time.sleep(5)
                        continue
                    
                    sim_result = sim_progress_resp.json()
                    status = sim_result.get("status")
                    logging.info(f"模拟 {sim_id} 状态: {status}")
                    
                    # 记录额外的调试信息
                    if status == "PENDING":
                        logging.debug(f"模拟 {sim_id} 仍在等待中...")
                    elif status == "RUNNING":
                        logging.debug(f"模拟 {sim_id} 正在运行...")
                    elif status not in ["COMPLETE", "ERROR"]:
                        logging.warning(f"模拟 {sim_id} 有未知状态: {status}")
                    
                    if status == "COMPLETE":
                        alpha_id = sim_result.get("alpha")
                        if alpha_id:
                            alpha_resp = self.sess.get(f'https://api.worldquantbrain.com/alphas/{alpha_id}')
                            if alpha_resp.status_code == 200:
                                alpha_data = alpha_resp.json()
                                fitness = alpha_data.get("is", {}).get("fitness")
                                logging.info(f"Alpha {alpha_id} 完成，适应度: {fitness}")
                                
                                self.results.append({
                                    "alpha": info["alpha"],
                                    "result": sim_result,
                                    "alpha_data": alpha_data
                                })
                                
                                # 检查适应度是否非空且大于阈值
                                if fitness is not None and fitness > 0.5:
                                    logging.info(f"发现有潜力的alpha！适应度: {fitness}")
                                    self.log_hopeful_alpha(info["alpha"], alpha_data)
                                    successful += 1
                                elif fitness is None:
                                    logging.warning(f"Alpha {alpha_id} 没有适应度数据，跳过有希望的alpha记录")
                    elif status == "ERROR":
                        logging.error(f"alpha模拟失败: {info['alpha']}")
                    completed.append(sim_id)
                    
                except Exception as e:
                    logging.error(f"检查 {sim_id} 结果时出错: {str(e)}")
        
        # 移除已完成的模拟
        for sim_id in completed:
            del self.pending_results[sim_id]
        
        # 将失败的模拟重新加入队列
        for alpha, sim_id in retry_queue:
            del self.pending_results[sim_id]
            self.retry_queue.add(alpha)
        
        return successful

    def test_alpha(self, alpha: str) -> Dict:
        """测试单个alpha表达式
        
        参数:
            alpha: 要测试的alpha表达式字符串
            
        返回:
            Dict: 测试结果，包含状态和消息或结果数据
        """
        result = self._test_alpha_impl(alpha)
        if result.get("status") == "error" and "SIMULATION_LIMIT_EXCEEDED" in result.get("message", ""):
            self.retry_queue.add(alpha)
            return {"status": "queued", "message": "已添加到重试队列"}
        return result

    def _test_alpha_impl(self, alpha_expression: str) -> Dict:
        """alpha测试的实现，包含适当的URL处理
        
        参数:
            alpha_expression: 要测试的alpha表达式字符串
            
        返回:
            Dict: 测试结果，包含状态和消息或结果数据
        """
        def submit_simulation():
            """提交模拟请求的内部函数"""
            simulation_data = {
                'type': 'REGULAR',  # 模拟类型为常规
                'settings': {
                    'instrumentType': 'EQUITY',  # 工具类型为股票
                    'region': 'USA',  # 地区为美国
                    'universe': 'TOP3000',  # 股票池为TOP3000
                    'delay': 1,  # 延迟设置
                    'decay': 0,  # 衰减设置
                    'neutralization': 'INDUSTRY',  # 行业中性化
                    'truncation': 0.08,  # 截断设置
                    'pasteurization': 'ON',  # 开启巴氏消毒（异常值处理）
                    'unitHandling': 'VERIFY',  # 单位验证
                    'nanHandling': 'OFF',  # 关闭NaN处理
                    'language': 'FASTEXPR',  # 使用FASTEXPR语言
                    'visualization': False,  # 不启用可视化
                },
                'regular': alpha_expression  # 要测试的alpha表达式
            }
            return self.sess.post('https://api.worldquantbrain.com/simulations', json=simulation_data)

        try:
            sim_resp = submit_simulation()
            
            # 处理认证错误
            if sim_resp.status_code == 401 or (
                sim_resp.status_code == 400 and 
                "authentication credentials" in sim_resp.text.lower()
            ):
                logging.warning("认证已过期，正在刷新会话...")
                self.setup_auth(self.credentials_path)  # 刷新认证
                sim_resp = submit_simulation()  # 使用新的认证重试
            
            # 检查响应状态码是否成功
            if sim_resp.status_code != 201:
                return {"status": "error", "message": sim_resp.text}

            # 从响应头获取进度URL
            sim_progress_url = sim_resp.headers.get('location')
            if not sim_progress_url:
                return {"status": "error", "message": "未收到进度URL"}

            # 返回成功结果，包含模拟ID和进度URL
            return {
                "status": "success", 
                "result": {
                    "id": f"{time.time()}_{random.random()}",  # 生成唯一的模拟ID
                    "progress_url": sim_progress_url  # 用于后续查询进度的URL
                }
            }
            
        except Exception as e:
            logging.error(f"测试alpha {alpha_expression} 时出错: {str(e)}")
            return {"status": "error", "message": str(e)}

    def log_hopeful_alpha(self, expression: str, alpha_data: Dict) -> None:
        """将有潜力的alpha记录到JSON文件
        
        参数:
            expression: alpha表达式字符串
            alpha_data: alpha的详细数据，包含性能指标和其他信息
        """
        log_file = 'hopeful_alphas.json'  # 日志文件路径
        
        # 加载现有数据
        existing_data = []
        if os.path.exists(log_file):
            try:
                with open(log_file, 'r') as f:
                    existing_data = json.load(f)
            except json.JSONDecodeError:
                print(f"警告: 无法解析 {log_file}，从头开始")
        
        # 添加新的alpha记录，包含时间戳
        entry = {
            "expression": expression,  # 存储表达式字符串
            "timestamp": int(time.time()),  # 添加时间戳
            "alpha_id": alpha_data.get("id", "unknown"),  # alpha的唯一标识符
            "fitness": alpha_data.get("is", {}).get("fitness"),  # 适应度值
            "sharpe": alpha_data.get("is", {}).get("sharpe"),  # 夏普比率
            "turnover": alpha_data.get("is", {}).get("turnover"),  # 换手率
            "returns": alpha_data.get("is", {}).get("returns"),  # 回报率
            "grade": alpha_data.get("grade", "UNKNOWN"),  # 评级
            "checks": alpha_data.get("is", {}).get("checks", [])  # 检查项结果
        }
        
        existing_data.append(entry)  # 添加新记录到现有数据
        
        # 保存更新后的数据
        with open(log_file, 'w') as f:
            json.dump(existing_data, f, indent=2)
        
        print(f"已将有潜力的alpha记录到 {log_file}")

    def get_results(self) -> List[Dict]:
        """获取所有处理过的结果，包括重试的alpha
        
        返回:
            List[Dict]: 包含所有处理结果的列表
        """
        return self.results

    def fetch_submitted_alphas(self):
        """从WorldQuant API获取已提交的alpha，包含重试逻辑
        
        返回:
            List[Dict]: 已提交的alpha列表，如果获取失败则返回空列表
        """
        url = "https://api.worldquantbrain.com/users/self/alphas"  # API endpoint
        params = {
            "limit": 100,  # 每页数量限制
            "offset": 0,  # 偏移量
            "status!=": "UNSUBMITTED%1FIS-FAIL",  # 排除未提交和失败的alpha
            "order": "-dateCreated",  # 按创建日期倒序排列
            "hidden": "false"  # 不包含隐藏的alpha
        }
        
        max_retries = 3  # 最大重试次数
        retry_delay = 60  # 重试间隔（秒）
        
        # 重试循环
        for attempt in range(max_retries):
            try:
                response = self.sess.get(url, params=params)
                
                # 处理速率限制（Too Many Requests）
                if response.status_code == 429:  # 太多请求
                    wait_time = int(response.headers.get('Retry-After', retry_delay))
                    logger.info(f"速率限制。等待 {wait_time} 秒后重试...")
                    time.sleep(wait_time)
                    continue
                    
                # 检查响应状态，如果不是200则抛出异常
                response.raise_for_status()
                return response.json()["results"]  # 返回结果数据
            
            except Exception as e:
                # 如果不是最后一次尝试，则重试
                if attempt < max_retries - 1:
                    logger.warning(f"尝试 {attempt + 1} 失败: {str(e)}。正在重试...")
                    time.sleep(retry_delay)
                else:
                    # 所有尝试都失败
                    logger.error(f"{max_retries} 次尝试后未能获取已提交的alpha: {e}")
                    return []
        
        return []  # 默认返回空列表

def extract_expressions(alphas):
    """从已提交的alpha中提取表达式
    
    参数:
        alphas: 已提交的alpha列表
        
    返回:
        List[Dict]: 包含表达式和性能指标的列表
    """
    expressions = []  # 存储提取的表达式
    for alpha in alphas:
        # 检查alpha是否有regular部分且包含code字段
        if alpha.get("regular") and alpha["regular"].get("code"):
            expressions.append({
                "expression": alpha["regular"]["code"],  # alpha表达式
                "performance": {
                    "sharpe": alpha["is"].get("sharpe", 0),  # 夏普比率
                    "fitness": alpha["is"].get("fitness", 0)  # 适应度值
                }
            })
    return expressions

def is_similar_to_existing(new_expression, existing_expressions, similarity_threshold=0.7):
    """检查新表达式是否与现有表达式过于相似
    
    参数:
        new_expression: 要检查的新表达式
        existing_expressions: 现有表达式列表
        similarity_threshold: 相似度阈值，默认为0.7
        
    返回:
        bool: 如果新表达式与现有表达式过于相似则返回True，否则返回False
    """
    for existing in existing_expressions:
        # 基本相似度检查
        if new_expression == existing["expression"]:
            return True
            
        # 检查结构相似度
        if structural_similarity(new_expression, existing["expression"]) > similarity_threshold:
            return True
    
    return False

def calculate_similarity(expr1: str, expr2: str) -> float:
    """使用基于标记的比较计算两个表达式之间的相似度
    
    参数:
        expr1: 第一个表达式
        expr2: 第二个表达式
        
    返回:
        float: 0到1之间的相似度值，0表示完全不同，1表示完全相同
    """
    # 规范化表达式并分词
    expr1_tokens = set(tokenize_expression(normalize_expression(expr1)))
    expr2_tokens = set(tokenize_expression(normalize_expression(expr2)))
    
    # 如果任一表达式没有有效标记，则相似度为0
    if not expr1_tokens or not expr2_tokens:
        return 0.0
    
    # 计算Jaccard相似度
    intersection = len(expr1_tokens.intersection(expr2_tokens))  # 交集大小
    union = len(expr1_tokens.union(expr2_tokens))  # 并集大小
    
    return intersection / union  # Jaccard系数 = 交集/并集

def structural_similarity(expr1, expr2):
    """计算两个表达式之间的结构相似度
    
    参数:
        expr1: 第一个表达式
        expr2: 第二个表达式
        
    返回:
        float: 0到1之间的相似度值
    """
    return calculate_similarity(expr1, expr2)  # 使用我们的相似度函数

def normalize_expression(expr):
    """规范化表达式以进行比较
    
    参数:
        expr: 要规范化的表达式
        
    返回:
        str: 规范化后的表达式
    """
    # 移除空白字符并转换为小写
    expr = re.sub(r'\s+', '', expr.lower())
    return expr

def tokenize_expression(expr):
    """将表达式分割成有意义的标记
    
    参数:
        expr: 要分词的表达式
        
    返回:
        List[str]: 标记列表
    """
    # 在操作符和括号处分割，同时保留它们
    tokens = re.findall(r'[\w._]+|[(),*/+-]', expr)  # 匹配标识符和操作符
    return tokens

def generate_alpha():
    """生成新的alpha表达式
    
    返回:
        str: 生成的唯一alpha表达式，如果无法生成则返回None
    """
    # 初始化alpha生成器
    generator = AlphaGenerator("./credential.txt", "http://localhost:11434")
    data_fields = generator.get_data_fields()  # 获取数据字段
    operators = generator.get_operators()  # 获取操作符
    
    # 首先获取已提交的alpha
    submitted_alphas = generator.fetch_submitted_alphas()
    existing_expressions = extract_expressions(submitted_alphas)
    
    max_attempts = 50  # 最大尝试次数
    attempts = 0  # 当前尝试次数
    
    while attempts < max_attempts:
        # 使用Ollama生成alpha想法
        alpha_ideas = generator.generate_alpha_ideas_with_ollama(data_fields, operators)
        for idea in alpha_ideas:
            # 检查是否与现有表达式相似
            if not is_similar_to_existing(idea, existing_expressions):
                logger.info(f"生成了唯一表达式: {idea}")
                return idea
                
        attempts += 1
        logger.debug(f"尝试 {attempts}: 所有表达式都太相似了")
    
    logger.warning("达到最大尝试次数后未能生成唯一表达式")
    return None

def main():
    """程序主入口，处理命令行参数并启动持续的alpha挖掘过程
    
    返回:
        int: 程序退出码，0表示正常退出，1表示遇到致命错误
    """
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='使用WorldQuant Brain API和Ollama/FinGPT生成和测试alpha因子')
    parser.add_argument('--credentials', type=str, default='./credential.txt',
                      help='凭据文件路径 (默认: ./credential.txt)')
    parser.add_argument('--output-dir', type=str, default='./results',
                      help='保存结果的目录 (默认: ./results)')
    parser.add_argument('--batch-size', type=int, default=3,
                      help='每批次生成的alpha因子数量 (默认: 3)')
    parser.add_argument('--sleep-time', type=int, default=10,
                      help='批次之间的睡眠时间（秒） (默认: 10)')
    parser.add_argument('--log-level', type=str, default='INFO',
                      choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                      help='设置日志级别 (默认: INFO)')
    parser.add_argument('--ollama-url', type=str, default='http://localhost:11434',
                      help='Ollama API URL (默认: http://localhost:11434)')
    parser.add_argument('--ollama-model', type=str, default='deepseek-r1:8b',
                                             help='要使用的Ollama模型 (默认: deepseek-r1:8b，适用于RTX A4000)')
    parser.add_argument('--max-concurrent', type=int, default=2,
                      help='最大并发模拟数量 (默认: 2)')
    
    args = parser.parse_args()
    
    # 配置日志
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),  # 输出到控制台
            logging.FileHandler('alpha_generator_ollama.log')  # 同时输出到文件
        ]
    )
    
    # 创建输出目录（如果不存在）
    os.makedirs(args.output_dir, exist_ok=True)
    
    try:
        # 初始化带Ollama的alpha生成器
        generator = AlphaGenerator(args.credentials, args.ollama_url, args.max_concurrent)
        generator.model_name = args.ollama_model  # 设置模型名称
        generator.initial_model = args.ollama_model  # 设置重置用的初始模型
        
        # 一次性获取数据字段和运算符
        print("正在获取数据字段和运算符...")
        data_fields = generator.get_data_fields()
        operators = generator.get_operators()
        
        batch_number = 1  # 批次数初始化为1
        total_successful = 0  # 成功alpha总数初始化为0
        
        print(f"开始持续alpha挖掘，批次大小: {args.batch_size}")
        print(f"结果将保存到: {args.output_dir}")
        print(f"使用Ollama服务: {args.ollama_url}")
        
        while True:
            try:
                logging.info(f"\n处理批次 #{batch_number}")
                logging.info("-" * 50)
                
                # 使用Ollama生成和提交批次
                alpha_ideas = generator.generate_alpha_ideas_with_ollama(data_fields, operators)
                batch_successful = generator.test_alpha_batch(alpha_ideas)
                total_successful += batch_successful  # 累加成功数量
                
                # 每处理几批进行一次VRAM清理
                generator.operation_count += 1
                if generator.operation_count % generator.vram_cleanup_interval == 0:
                    generator.cleanup_vram()
                
                # 保存批次结果
                results = generator.get_results()
                timestamp = int(time.time())  # 获取当前时间戳
                output_file = os.path.join(args.output_dir, f'batch_{batch_number}_{timestamp}.json')
                with open(output_file, 'w') as f:
                    json.dump(results, f, indent=2)
                
                logging.info(f"批次 {batch_number} 结果已保存至 {output_file}")
                logging.info(f"本批次成功数量: {batch_successful}")
                logging.info(f"累计成功alpha数量: {total_successful}")
                
                batch_number += 1  # 批次数增加1
                
                # 批次之间睡眠指定时间
                print(f"睡眠 {args.sleep_time} 秒...")
                sleep(args.sleep_time)
                
            except Exception as e:  # 处理批次级错误
                logging.error(f"批次 {batch_number} 发生错误: {str(e)}")
                logging.info("5分钟后重试...")
                sleep(300)
                continue
        
    except KeyboardInterrupt:  # 处理用户中断
        logging.info("\n正在停止alpha挖掘...")
        logging.info(f"总共处理批次: {batch_number - 1}")
        logging.info(f"累计成功alpha数量: {total_successful}")
        return 0
        
    except Exception as e:  # 处理致命错误
        logging.error(f"致命错误: {str(e)}")
        return 1

if __name__ == "__main__":
    exit(main())
