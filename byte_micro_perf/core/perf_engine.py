# Copyright 2023 ByteDance and/or its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import os
import sys
import csv
import json
import argparse
import pathlib
import itertools
import jsonlines
import shutil
import psutil
import prettytable
from datetime import timedelta

from typing import Any, Dict, List
import itertools
from collections import namedtuple

import traceback
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

FILE_DIR = pathlib.Path(__file__).parent.absolute()
MICRO_PERF_DIR = FILE_DIR.parent

sys.path.insert(0, str(MICRO_PERF_DIR))

from core.utils import logger, setup_logger
from core.scheduler import Scheduler

def _generate_profiler_dir_name(self, op_instance, process_id):
    """
    生成性能分析目录名称
    
    Args:
        op_instance: 操作实例
        process_id: 进程ID
        
    Returns:
        str: 生成的目录名称
    """
    case_info = ""
    
    # 尝试从op_instance获取测试用例文件名
    if hasattr(op_instance, 'args_dict') and '__file__' in op_instance.args_dict:
        # 从参数中提取文件名（去掉路径和扩展名）
        testcase_file = pathlib.Path(op_instance.args_dict['__file__']).stem
        case_info += f"{testcase_file}_"
    elif hasattr(op_instance, '__class__'):
        # 使用操作类名作为备选
        op_name = op_instance.__class__.__name__
        # 移除可能的后缀如Op等，使名称更简洁
        op_name = op_name.replace('Op', '').lower()
        case_info += f"{op_name}_"
    
    if hasattr(op_instance, 'args_dict'):
        args = op_instance.args_dict
        
        # 添加测试用例索引（如果存在）
        if 'index' in args:
            case_info += f"case_{args['index']}_"
        
        # 添加关键参数以区分测试用例
        # dtype相关参数
        for dtype_key in ['dtype', 'dst_dtype']:
            if dtype_key in args:
                case_info += f"{dtype_key}_{args[dtype_key]}_"
        
        # 大小相关参数
        for size_key in ['batch_size', 'num_tokens', 'hidden_size']:
            if size_key in args:
                case_info += f"{size_key}_{args[size_key]}_"
        
        # 模式相关参数（针对LLM测试）
        if 'mode' in args:
            case_info += f"mode_{args['mode']}_"
        
        # 维度相关参数
        if 'dim_size' in args:
            case_info += f"dim_{args['dim_size']}_"
        
        # head相关参数（针对LLM测试）
        for head_key in ['q_head_num', 'kv_head_num', 'head_dim']:
            if head_key in args:
                case_info += f"{head_key}_{args[head_key]}_"
        
        # world_size参数（针对分布式操作）
        if 'world_size' in args:
            case_info += f"ws_{args['world_size']}_"
    
    # 清理末尾的下划线
    case_info = case_info.rstrip('_')
    
    # 如果没有提取到参数信息，则使用默认命名
    if case_info:
        return f"{case_info}_{process_id}"
    else:
        return f"{process_id}"

def parse_task(task_dir, task):
    task_dir = pathlib.Path(task_dir).absolute()
    target_task_files = task_dir.rglob(task + ".json")
    task_cases = []
    for task_file in target_task_files:
        with open(task_file, "r") as f:
            json_data = json.load(f)
        json_data_list = json_data.get("cases", [])
        if json_data_list:
            for argument_case in json_data_list:
                removed_keys = []
                for key in argument_case:
                    if key.startswith("__"):
                        removed_keys.append(key)
                for removed_key in removed_keys:
                    argument_case.pop(removed_key)

                # 添加文件路径信息到参数中（只保留文件名，不包括后缀）
                argument_case["__file__"] = task_file.stem

                keys = list(argument_case.keys())
                values = list(argument_case.values())
                for i, value in enumerate(values):
                    if isinstance(value, str):
                        values[i] = [value]                
                total_cases = list(itertools.product(*values))
                for case in total_cases:
                    case_dict = dict(zip(keys, case))

                    added_key_value = {}
                    removed_key = []
                    for key in case_dict:
                        if "." in key:
                            split_keys = key.split(".")
                            for i, split_key in enumerate(split_keys):
                                added_key_value[split_key] = case_dict[key][i]
                            removed_key.append(key)
                    for key in removed_key:
                        del case_dict[key]
                    case_dict.update(added_key_value)
                    task_cases.append(case_dict)
    return task_cases



def engine_run(rank, *args):
    world_size, queue_instance, configs = args
    _ = queue_instance.get()

    # assign node
    node_world_size = configs.node_world_size
    node_rank = configs.node_rank

    # assign numa
    numa_world_size = world_size
    numa_rank = rank
    configs.numa_world_size = numa_world_size
    configs.numa_rank = numa_rank

    # assign all process
    all_process_size = node_world_size * numa_world_size
    all_process_rank = node_rank * numa_world_size + numa_rank
    configs.all_process_size = all_process_size
    configs.all_process_rank = all_process_rank

    # try init process dist
    os.environ['MASTER_PORT'] = os.environ['GLOO_PORT']
    try:
        dist.init_process_group(
            backend="gloo", 
            world_size=all_process_size,
            rank=all_process_rank, 
            timeout=timedelta(seconds=1800)
        )
        data = torch.ones(1)
        dist.all_reduce(data)
    except Exception as e:
        traceback.print_exc()
        sys.exit(1)
        
    # try create scheduler
    try:
        scheduler = Scheduler(configs)
    except Exception as e:
        traceback.print_exc()
        sys.exit(-1)

    for i, task in enumerate(configs.task.split(",")):
        # sync for all processes
        dist.barrier()
        if numa_rank == 0:
            print("")
            print("=" * 100)
            print(f"{i+1}: {task}")
            print("=" * 100)

        try:
            # parse task simutaneously
            task_cases = parse_task(configs.task_dir, task)
            if len(task_cases) == 0:
                print(f"{task} has no task case")
                continue

            # prepare task, exit if needed
            result_list = []
            prepare_ret = scheduler.prepare_task(task, task_cases)
            if prepare_ret:
                result_list = scheduler.run()

            if configs.numa_rank == 0:
                if len(result_list) == 0:
                    logger.error("No result available")
                    continue

                sku_name = result_list[0]["sku_name"]
                report_dir = pathlib.Path(configs.report_dir).absolute()
                backend_dir = report_dir.joinpath(configs.hardware_type)
                sku_dir = backend_dir.joinpath(sku_name)
                op_dir = sku_dir.joinpath(task)
                op_dir.mkdir(parents=True, exist_ok=True)

                # grouped by arg_type
                result_mapping = {}
                for result in result_list:
                    # check arguments and targets
                    arguments = result.get("arguments", {})
                    targets = result.get("targets", {})
                    if arguments == {} or targets == {}:
                        continue
                    
                    arg_type = arguments.get("arg_type", "default")
                    if arg_type not in result_mapping:
                        result_mapping[arg_type] = {}

                    provider = result.get("provider", "torch")
                    world_size = arguments.get("world_size", 1)
                    dtype = arguments["dtype"]

                    key = (provider, world_size, dtype)
                    if key not in result_mapping[arg_type]:
                        result_mapping[arg_type][key] = []
                    result_mapping[arg_type][key].append(result)
                


                for arg_type in result_mapping:
                    target_folder = op_dir.joinpath(arg_type)
                    if target_folder.exists():
                        shutil.rmtree(target_folder)
                    target_folder.mkdir(parents=True, exist_ok=True)

                    for key in result_mapping[arg_type]:
                        provider, world_size, dtype = key

                        if world_size == 1:
                            file_name = f"{provider}-{dtype}"
                        else:
                            file_name = f"{provider}-group{world_size}-{dtype}"

                        result_list = result_mapping[arg_type][key]
                        jsonl_file_path = target_folder.joinpath(file_name + ".jsonl")
                        with jsonlines.open(jsonl_file_path, "w") as writer:
                            for result in result_list:
                                writer.write(result)

                        keys = ["task_name"]
                        keys.extend(list(result_list[0]["arguments"].keys()))
                        keys.extend(list(result_list[0]["targets"].keys()))

                        csv_file_path = target_folder.joinpath(file_name + ".csv")
                        with open(csv_file_path, "w") as f:
                            writer = csv.writer(f)
                            writer.writerow(keys)
                            for result in result_list:
                                row = [task]
                                row.extend(list(result["arguments"].values()))
                                row.extend(list(result["targets"].values()))
                                writer.writerow(row)

        except Exception as e:
            traceback.print_exc()
            continue

    dist.destroy_process_group()
