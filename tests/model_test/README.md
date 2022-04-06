# 400T模型集成测试


## 目录结构
- models.json： 包含所有CI测试模型配置信息

示例如下，frontend段包含前端配置信息, quantization段包含量化所需配置信息; path指向demodels-lfs/$framework/下的相对路径
```json
"resnet50-v1": {
    "framework": "onnx",
    "path": "resnet50-v1/resnet50-v1-7.onnx",
    "input_names": ["data"],
    "input_shapes": [[1,3,224,224]],
    "input_dtypes": ["float32"],
    "frontend": {
        "layout": "NCHW",
        "preproc_method": "mean_scale",
        "means": [0,0,0],
        "scales": [255,255,255]
    },
    "quantization": {
        "mean": [123.675,116.28,103.53],
        "std": [58.395,57.12,57.375],
        "axis": 1
    },
    "tag": ["classification"],
    "backend": {}
}
```

- tasks.py：包含各个测试步骤具体实现, 每个步骤通过启动子进程调用，用来隔离不同步骤的环境、日志等；通过命令行传递测试参数

- task_ci_integration.py：集成测试入口脚本

## CI集成说明

CI job对脚本中model list包含的全部模型进行测试
```
python tests/model_test/task_ci_integration.py
```

模型批量测试结果包含在job artifact中，需要从gitlab job页面下载到本地查看

### 自定义CI测试配置
下述变量可由用户在运行自定义ci schedule pipeline时设置

- TVM_CI_INTEGRATION_MODEL_LIST： 要测试的模型名称列表，逗号分隔，需要在models.json加入对应配置

- TVM_CI_PARALLELISM： 测试并行度，默认2进程并发

- TVM_CI_WORKING_DIR:  工作目录和运行日志保存路径，本地测试时可调整

- TVM_CI_DEMODELS_DIR： DEMODELS项目根目录，本地测试时可调整
