import contextlib
import os
import time
import argparse
import threading
import concurrent.futures
import queue
import json
import requests
from typing import List, Dict, Any
import statistics

# Third Party
from transformers import AutoTokenizer
import shutil


def setup_environment_variables():
    """设置环境变量"""
    os.environ["PYTHONHASHSEED"] = "123456"


def read_jsonl(file_path):
    """读取JSONL文件"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:  # 跳过空行
                data.append(json.loads(line))
    return data


def get_model_name_from_path(model_path: str) -> str:
    """从模型路径中提取模型名称"""
    model_path = model_path.rstrip('/')
    model_name = os.path.basename(model_path)
    return model_name


class RequestStats:
    """记录单个请求的统计信息"""
    def __init__(self, request_id: str, dataset: str, prompt_length: int):
        self.request_id = request_id
        self.dataset = dataset
        self.prompt_length = prompt_length
        self.start_time = None
        self.end_time = None
        self.generated_text = ""
        self.tokens_used = 0
        
    def start(self):
        self.start_time = time.perf_counter()
        
    def end(self, generated_text: str, tokens_used: int = 0):
        self.end_time = time.perf_counter()
        self.generated_text = generated_text
        self.tokens_used = tokens_used
        
    @property
    def latency(self) -> float:
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return 0.0
    
    @property
    def tps(self) -> float:
        """计算该请求的TPS (Tokens Per Second)"""
        if self.latency > 0 and self.tokens_used > 0:
            return self.tokens_used / self.latency
        return 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "request_id": self.request_id,
            "dataset": self.dataset,
            "prompt_length": self.prompt_length,
            "latency": self.latency,
            "tps": self.tps,
            "tokens_used": self.tokens_used,
            "generated_text_length": len(self.generated_text)
        }


class PerformanceMonitor:
    """性能监控器，记录所有请求的统计信息"""
    def __init__(self):
        self.stats_lock = threading.Lock()
        self.all_stats: List[RequestStats] = []
        self.start_time = None
        self.end_time = None
        
    def start_monitoring(self):
        self.start_time = time.perf_counter()
        
    def stop_monitoring(self):
        self.end_time = time.perf_counter()
        
    def add_request_stats(self, stats: RequestStats):
        with self.stats_lock:
            self.all_stats.append(stats)
            
    def get_dataset_stats(self, dataset: str) -> List[RequestStats]:
        with self.stats_lock:
            return [stats for stats in self.all_stats if stats.dataset == dataset]
            
    def get_overall_tps(self) -> float:
        """计算整体TPS"""
        if not self.all_stats or not self.start_time or not self.end_time:
            return 0.0
            
        total_time = self.end_time - self.start_time
        if total_time <= 0:
            return 0.0
            
        total_tokens = sum(stats.tokens_used for stats in self.all_stats)
        return total_tokens / total_time
    
    def save_performance_report(self, output_path: str):
        """保存性能报告"""
        report = {
            "overall": {
                "total_requests": len(self.all_stats),
                "total_time": self.end_time - self.start_time if self.end_time else 0,
                "overall_tps": self.get_overall_tps(),
                "average_latency": sum(stats.latency for stats in self.all_stats) / len(self.all_stats) if self.all_stats else 0,
                "average_tps": sum(stats.tps for stats in self.all_stats) / len(self.all_stats) if self.all_stats else 0,
                "total_tokens_used": sum(stats.tokens_used for stats in self.all_stats),
            },
            "by_dataset": {},
            "detailed_requests": [stats.to_dict() for stats in self.all_stats]
        }
        
        # 按数据集统计
        datasets = set(stats.dataset for stats in self.all_stats)
        for dataset in datasets:
            dataset_stats = self.get_dataset_stats(dataset)
            if dataset_stats:
                report["by_dataset"][dataset] = {
                    "request_count": len(dataset_stats),
                    "average_latency": sum(stats.latency for stats in dataset_stats) / len(dataset_stats),
                    "average_tps": sum(stats.tps for stats in dataset_stats) / len(dataset_stats),
                    "total_tokens_used": sum(stats.tokens_used for stats in dataset_stats),
                }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)


class OnlineInferenceClient:
    """在线推理客户端"""
    
    def __init__(self, api_url: str, model_name: str, max_workers: int = 10):
        self.api_url = api_url
        self.model_name = model_name
        self.max_workers = max_workers
        self.lock = threading.Lock()
        
    def call_model(self, prompt: str, request_id: str, sampling_params: Dict = None) -> tuple:
        """
        调用在线推理服务
        Returns: (response_text, time_taken, tokens_used)
        """
        if sampling_params is None:
            sampling_params = {
                'temperature': 0,
                'top_p': 0.95,
                'max_tokens': 1024,
                'ignore_eos': False
            }
        
        headers = {
            'Content-Type': 'application/json'
        }
        
        # vLLM API payload format
        payload = {
            'model': self.model_name,
            'prompt': prompt,
            'temperature': sampling_params['temperature'],
            'top_p': sampling_params['top_p'],
            'max_tokens': sampling_params['max_tokens'],
            'ignore_eos': sampling_params['ignore_eos'],
            'stream': False
        }
        
        start_time = time.perf_counter()
        try:
            response = requests.post(self.api_url, json=payload, headers=headers)
            response.raise_for_status()
            result = response.json()
            end_time = time.perf_counter()
            
            # Extract response content from vLLM format
            content = result['choices'][0]['text']
            
            # Get token usage
            tokens_used = result.get('usage', {}).get('total_tokens', len(content) // 4)
            
            return content, end_time - start_time, tokens_used
            
        except requests.exceptions.Timeout:
            with self.lock:
                print(f"Request {request_id} timed out")
            return "", 0, 0
        except Exception as e:
            with self.lock:
                print(f"API call error for request {request_id}: {e}")
            if 'response' in locals():
                with self.lock:
                    print(f"Response: {response.text}")
            return "", 0, 0


def process_single_request(client: OnlineInferenceClient, prompt: str, sampling_params: Dict,
                          request_id: str, dataset: str, performance_monitor: PerformanceMonitor,
                          result_queue: queue.Queue, ucm_examples_dir: str):
    """处理单个请求"""
    # 清理数据目录（模拟UCM环境准备）
    #data_dir = os.path.join(ucm_examples_dir, "data")
    #shutil.rmtree(data_dir, ignore_errors=True)
    #os.makedirs(data_dir, exist_ok=True)
    
    stats = RequestStats(request_id, dataset, len(prompt))
    stats.start()
    
    try:
        response_text, time_taken, tokens_used = client.call_model(
            prompt, request_id, sampling_params
        )
        
        stats.end(response_text, tokens_used)
        performance_monitor.add_request_stats(stats)
        
        # 将结果放入队列
        result_queue.put({
            "request_id": request_id,
            "dataset": dataset,
            "stats": stats,
            "generated_text": response_text
        })
        
        with client.lock:
            print(f"Request {request_id} completed in {stats.latency:.2f}s, TPS: {stats.tps:.2f}")
        
    except Exception as e:
        with client.lock:
            print(f"Request {request_id} failed: {str(e)}")
        # 即使失败也记录统计信息
        stats.end("", 0)
        performance_monitor.add_request_stats(stats)


def run_dataset_inference(client: OnlineInferenceClient, dataset_name: str, jsonl_data: List[Dict], 
                         sampling_params: Dict, output_path: str,
                         performance_monitor: PerformanceMonitor, ucm_examples_dir: str,
                         max_workers: int = 10, test_num: int = 200, 
                         test_max_len: int = 69000):
    """运行数据集的推理测试"""
    
    # 准备请求数据
    requests = []
    for i, item in enumerate(jsonl_data):
        if i >= test_num:
            break
            
        answer = item.get("answers", "")
        length = len(item["context"])
        
        # 根据数据集构建不同的prompt
        if dataset_name == "multifieldqa_en":
            prompt = f"""Read the following text and answer briefly in English:\n\n{item["context"]}\n\nNow based on the article above, answer the question below. Only provide the answer, do not output any other words.\n\nQuestion: {item["input"]}\nAnswer:"""
        elif dataset_name == "qasper":
            prompt = f"""Read the following academic paper excerpt and answer the question based on the content. Provide a concise and accurate answer.\n\nPaper Excerpt:\n{item["context"]}\n\nQuestion: {item["input"]}\nAnswer:"""
        elif dataset_name == "2wiki":
            prompt = f"""Based on the given Wikipedia articles, answer the following question. You may need to combine information from multiple parts of the text to find the correct answer.\n\nArticles:\n{item["context"]}\n\nQuestion: {item["input"]}\nAnswer:"""
        elif dataset_name == "trec":
            prompt = f"""Classify the following question into one of the following categories:{item["context"]} {item["input"]}"""
        elif dataset_name == "lcc":
            prompt = f"""Read the following content and generate next line code:{item["context"]} Only answer with one line of code"""
        else:
            prompt = f"{item['context']}\n\nQuestion: {item['input']}\nAnswer:"
            
        requests.append({
            "request_id": f"{dataset_name}_{i}",
            "prompt": prompt,
            "item": item,
            "answer": answer,
            "length": length
        })
    
    # 创建结果队列
    result_queue = queue.Queue()
    
    print(f"Starting {dataset_name} with {len(requests)} requests, max_workers: {max_workers}")
    
    # 使用线程池并发执行请求
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务
        future_to_request = {}
        for req in requests:
            future = executor.submit(
                process_single_request,
                client, req["prompt"], sampling_params, 
                req["request_id"], dataset_name, performance_monitor,
                result_queue, ucm_examples_dir
            )
            future_to_request[future] = req
            
        # 等待所有任务完成
        completed = 0
        for future in concurrent.futures.as_completed(future_to_request):
            completed += 1
            req = future_to_request[future]
            try:
                future.result()
            except Exception as exc:
                print(f'Request {req["request_id"]} generated an exception: {exc}')
            
            if completed % 10 == 0:
                print(f"Completed {completed}/{len(requests)} requests for {dataset_name}")
    
    # 处理结果并保存
    processed_results = []
    while not result_queue.empty():
        try:
            result = result_queue.get_nowait()
            processed_results.append(result)
        except queue.Empty:
            break
    
    # 按请求ID排序结果以确保顺序
    processed_results.sort(key=lambda x: x["request_id"])
    
    # 保存推理结果
    with open(output_path, 'w', encoding='utf-8') as f:
        for result in processed_results:
            req_id = result["request_id"]
            original_req = next(req for req in requests if req["request_id"] == req_id)
            
            output_data = {
                "pred": result["generated_text"],
                "answers": original_req["answer"],
                "length": original_req["length"],
                "E2E_time": result["stats"].latency,
                "tps": result["stats"].tps,
                "tokens_used": result["stats"].tokens_used
            }
            json.dump(output_data, f, ensure_ascii=False)
            f.write('\n')
    
    print(f"Completed {dataset_name} with {len(processed_results)} requests")


def main():
    parser = argparse.ArgumentParser(description="Run online LLM inference benchmark")
    parser.add_argument("--api_url", type=str, required=True, help="vLLM API endpoint URL")
    parser.add_argument("--model_name", type=str, required=True, help="Model name for inference")
    parser.add_argument("--ucm_examples_dir", type=str, required=True, help="Path to UCM examples directory")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to data directory")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to output directory")
    parser.add_argument("--max_workers", type=int, default=10, help="Maximum number of concurrent requests (5-30)")
    parser.add_argument("--test_num", type=int, default=200, help="Number of test cases per dataset")
    
    args = parser.parse_args()

    # 验证并发数范围
    if args.max_workers < 5 or args.max_workers > 80:
        print(f"Warning: max_workers {args.max_workers} is outside recommended range 5-80, using default 10")
        args.max_workers = 10

    test_max_len = 69000
    setup_environment_variables()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 创建性能监控器
    performance_monitor = PerformanceMonitor()
    
    # 创建在线推理客户端
    client = OnlineInferenceClient(args.api_url, args.model_name, args.max_workers)
    
    # 记录整体开始时间
    overall_start_time = time.time()
    performance_monitor.start_monitoring()
    
    # 定义数据集配置
    datasets_config = [
        {
            "name": "multifieldqa_en",
            "file": "multifieldqa_en.jsonl",
            "sampling_params": {
                'temperature': 0,
                'top_p': 0.95,
                'max_tokens': 1024,
                'ignore_eos': False
            }
        },
        {
            "name": "qasper", 
            "file": "qasper.jsonl",
            "sampling_params": {
                'temperature': 0,
                'top_p': 0.95,
                'max_tokens': 1024,
                'ignore_eos': False
            }
        },
        {
            "name": "2wiki",
            "file": "2wikimqa.jsonl", 
            "sampling_params": {
                'temperature': 0,
                'top_p': 0.95,
                'max_tokens': 1024,
                'ignore_eos': False
            }
        },
        {
            "name": "trec",
            "file": "trec.jsonl",
            "sampling_params": {
                'temperature': 0,
                'top_p': 0.95,
                'max_tokens': 1024,
                'ignore_eos': False
            }
        },
        {
            "name": "lcc",
            "file": "lcc.jsonl",
            "sampling_params": {
                'temperature': 0,
                'top_p': 0.95,
                'max_tokens': 1024,
                'ignore_eos': False
            }
        }
    ]
    
    # 运行所有数据集的测试
    for dataset_config in datasets_config:
        print(f"\n{'='*60}")
        print(f"Starting inference for {dataset_config['name']}")
        print(f"{'='*60}")
        
        try:
            data_dir = os.path.join(args.ucm_examples_dir, "data")
            shutil.rmtree(data_dir, ignore_errors=True)
            os.makedirs(data_dir, exist_ok=True)
            jsonl_data = read_jsonl(os.path.join(args.data_dir, dataset_config["file"]))
            output_path = os.path.join(args.output_dir, f"{args.model_name}_{dataset_config['name']}_results.jsonl")
            
            run_dataset_inference(
                client=client,
                dataset_name=dataset_config["name"],
                jsonl_data=jsonl_data,
                sampling_params=dataset_config["sampling_params"],
                output_path=output_path,
                performance_monitor=performance_monitor,
                ucm_examples_dir=args.ucm_examples_dir,
                max_workers=args.max_workers,
                test_num=args.test_num,
                test_max_len=test_max_len
            )
        except Exception as e:
            print(f"Error processing dataset {dataset_config['name']}: {e}")
            continue
    
    performance_monitor.stop_monitoring()
    
    # 记录整体结束时间
    overall_end_time = time.time()
    overall_total_time = overall_end_time - overall_start_time
    
    # 保存性能报告
    performance_report_path = os.path.join(args.output_dir, f"{args.model_name}_performance_report.json")
    performance_monitor.save_performance_report(performance_report_path)
    
    # 打印总结报告
    print(f"\n{'='*60}")
    print("BENCHMARK SUMMARY")
    print(f"{'='*60}")
    print(f"Model: {args.model_name}")
    print(f"API Endpoint: {args.api_url}")
    print(f"Concurrent Workers: {args.max_workers}")
    print(f"Total Requests: {len(performance_monitor.all_stats)}")
    print(f"Overall TPS: {performance_monitor.get_overall_tps():.2f}")
    print(f"Average Latency: {sum(stats.latency for stats in performance_monitor.all_stats) / len(performance_monitor.all_stats):.2f}s")
    print(f"Total Tokens Used: {sum(stats.tokens_used for stats in performance_monitor.all_stats)}")
    print(f"Benchmark Execution Time: {overall_total_time:.2f}s")
    print(f"Performance report saved to: {performance_report_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()