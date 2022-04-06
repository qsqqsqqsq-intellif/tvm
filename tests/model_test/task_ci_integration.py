# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
# encoding=utf-8
"""CI模型集成测试"""
import os
import json
import multiprocessing
import subprocess
import tempfile
import time
import tvm

# 目标测试模型list
DEFAULT_TEST_MODELS = "resnet34-v1,resnet50-v1"
TEST_MODELS = os.environ.get("TVM_CI_INTEGRATION_MODEL_LIST", DEFAULT_TEST_MODELS)
TEST_MODELS = [_.strip() for _ in TEST_MODELS.split(",") if _.strip() != ""]
print(f"【测试模型列表】{TEST_MODELS}")

# 加载模型配置文件
TVM_HOME = os.path.abspath(os.environ.get("TVM_HOME", f"{tvm.__path__[0]}/../.."))
MODEL_TEST_HOME = os.path.join(TVM_HOME, "tests/model_test")
with open(os.path.join(MODEL_TEST_HOME, "models.json")) as fp:
    MODEL_CONFIGS = json.load(fp)

WORKER_NUM = int(os.environ.get("TVM_CI_PARALLELISM", "2"))
print(f"【模型测试并发度】{WORKER_NUM}")

# 测试日志目录
WORKING_DIR = os.path.join(os.environ.get("TVM_CI_WORKING_DIR", tempfile.mkdtemp()), "integration")
print(f"【测试工作目录】{WORKING_DIR}")

# DEMODELS目录
DEMODELS_DIR = os.environ.get("TVM_CI_DEMODELS_DIR", "/data/share/demodels-lfs")
print(f"【DEMODELS目录】{DEMODELS_DIR}")


# 启动单独测试任务进程
def launch_task(task_config, model_config, working_dir, task_records, timeout=3000):
    args = ["python3", "./tasks.py"]
    title = task_config.pop("title")
    model_name = task_config["model-name"]
    log_file = os.path.join(working_dir, task_config["task"] + ".log")
    log_file_fd = open(log_file, "w")
    envs = {k: v for k, v in os.environ.items()}
    if "environments" in task_config:
        extra_envs = task_config.pop("environments")
        for k in extra_envs:
            envs[k] = extra_envs[k]
    for k in task_config:
        args.append(f"--{k}={task_config[k]}")
    proc = subprocess.Popen(
        args, cwd=MODEL_TEST_HOME, stdout=log_file_fd, stderr=log_file_fd, env=envs
    )
    st = time.time()
    proc.wait(timeout)
    et = time.time()
    record = {
        "task": task_config["task"],
        "model": model_name,
        "time": et - st,
        "exitcode": proc.returncode,
        "log": os.path.relpath(log_file, WORKING_DIR),
        "result": os.path.join(working_dir, "result.txt"),
        "framework": model_config["framework"],
    }
    task_records.append(record)
    with open(os.path.join(working_dir, task_config["task"] + ".record.json"), "w") as outf:
        json.dump(record, outf, indent=4)
    if proc.returncode != 0:
        print(f"【{model_name}】: {title}失败, exitcode={proc.returncode}, log={log_file}")
        return False
    print(f"【{model_name}】: {title}成功")
    return True


# 端到端测试流程
def run_network(model_name, result_records):
    model_config = MODEL_CONFIGS[model_name]
    print(f"【模型配置】【{model_name}】{model_config}")
    working_dir = os.path.join(WORKING_DIR, model_name)
    if not os.path.isdir(working_dir):
        os.makedirs(working_dir)

    # task1: 测试前端转换模型graph_executor
    task_config = {
        "task": "convert_frontend_model",
        "output-dir": os.path.join(working_dir, "converted"),
        "input-file": os.path.join(DEMODELS_DIR, model_config["framework"], model_config["path"]),
        "model-name": model_name,
        "title": "前端模型转换",
    }
    convert_success = launch_task(task_config, model_config, working_dir, result_records)
    if not convert_success:
        return

    # task2: 前端模型和源框架浮点比对
    task_config = {
        "task": "verify_frontend_model",
        "json": os.path.join(working_dir, "converted", model_name + ".json"),
        "params": os.path.join(working_dir, "converted", model_name + ".params"),
        "input-file": os.path.join(DEMODELS_DIR, model_config["framework"], model_config["path"]),
        "model-name": model_name,
        "title": "前端框架浮点比对",
    }
    launch_task(task_config, model_config, working_dir, result_records)

    # task3: 量化模型
    task_config = {
        "task": "quantize_model",
        "output-dir": os.path.join(working_dir, "quantized"),
        "json": os.path.join(working_dir, "converted", model_name + ".json"),
        "params": os.path.join(working_dir, "converted", model_name + ".params"),
        "model-name": model_name,
        "title": "模型量化",
    }
    quantize_success = launch_task(task_config, model_config, working_dir, result_records)
    if not quantize_success:
        return

    # task4: nnp模型编译
    task_config = {
        "task": "compile_nnp_model",
        "output-dir": working_dir,
        "json": os.path.join(working_dir, "quantized", model_name + ".json"),
        "params": os.path.join(working_dir, "quantized", model_name + ".params"),
        "model-name": model_name,
        "title": "nnp模型编译",
    }
    compile_success = launch_task(task_config, model_config, working_dir, result_records)
    if not compile_success:
        return

    # task5: nnp iss运行测试
    task_config = {
        "task": "run_and_check_iss",
        "input-file": os.path.join(working_dir, model_name + ".so"),
        "json": os.path.join(working_dir, "quantized", model_name + ".json"),
        "params": os.path.join(working_dir, "quantized", model_name + ".params"),
        "model-name": model_name,
        "title": "iss计算结果比对",
        "environments": {"EDGEX_DEBUG_ISS": "on"},
    }
    launch_task(task_config, model_config, working_dir, result_records)


# 收集测试日志
def make_summary(all_records):
    all_tables = {}
    task_names = [
        "convert_frontend_model",
        "verify_frontend_model",
        "quantize_model",
        "compile_nnp_model",
        "run_and_check_iss",
    ]
    # 按 框架 - 模型名 - 测试点 整理测试记录
    for record_list in all_records:
        for record in record_list:
            if record["framework"] not in all_tables:
                all_tables[record["framework"]] = {}
            table = all_tables[record["framework"]]
            if record["model"] not in table:
                table[record["model"]] = {}
            item = table[record["model"]]
            if record["exitcode"] == 0:
                summary = (
                    f"<font color=\"green\">【成功】</font><br>运行时间: {record['time']:.1f} sec"
                    + f"<br>日志: <a href=\"{record['log']}\">运行日志</a>"
                )
                if os.path.isfile(record["result"]):
                    with open(record["result"], "r") as infile:
                        res = infile.read().strip()
                        if len(res) > 0:
                            summary += "<br>验证结果: " + res
            else:
                summary = (
                    f"<font color=\"red\">【失败】</font><br>运行时间: {record['time']:.1f} sec"
                    + f"<br>日志: <a href=\"{record['log']}\">运行日志</a>"
                )
            item[record["task"]] = summary

    # 写入html表格文件
    html_template = """
    <html>
        <table border="1" cellpadding="0" cellspacing="0" width=800 padding-top=5 padding-bottom=5>
        <tr>
            <th>模型名称</th>
            <th>前端转化</th>
            <th>前端浮点精度比对</th>
            <th>模型量化</th>
            <th>模型编译</th>
            <th>ISS结果比对</th>
        </tr>
        {ROWS}
        </table>
    </html>
    """
    for framework in all_tables:
        table = all_tables[framework]
        rows = ""
        for model_name in sorted(list(table)):
            row = [model_name] + ["" for _ in task_names]
            for task_key in table[model_name]:
                idx = 1 + task_names.index(task_key)
                row[idx] = table[model_name][task_key]
            row_html = "<tr>"
            for data in row:
                row_html += f"<td>{data}</td>"
            row_html += "</tr>\n"
            rows += row_html
        html = html_template.replace("{ROWS}", rows)
        with open(os.path.join(WORKING_DIR, framework + ".html"), "w") as outf:
            outf.write(html)


def run_sub_worker(model_list, records):
    for model_name in model_list:
        try:
            run_network(model_name, records)
        except Exception as e:
            print(e)  # continue to skip error


def run_integration():
    all_records = []
    workers = []
    workload_cnt = (len(TEST_MODELS) + WORKER_NUM - 1) // WORKER_NUM
    for i in range(WORKER_NUM):
        all_records.append([])
        p = multiprocessing.Process(
            target=run_sub_worker,
            args=(TEST_MODELS[i * workload_cnt : i * workload_cnt + workload_cnt], all_records[i]),
            daemon=True,
        )
        workers.append(p)
        p.start()
    for i in range(WORKER_NUM):
        workers[i].join()
    make_summary(all_records)


if __name__ == "__main__":
    run_integration()
