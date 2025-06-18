# yolov8-
yolov8剪枝，模型轻量化的实现】

### 剪枝概述

模型剪枝是一种优化技术，旨在减少神经网络中不必要的参数，从而降低模型的复杂性和计算负载，提高模型的效率。对于YOLOv8模型，剪枝可以显著降低计算复杂度和内存占用，同时保持较高的精度。

### 剪枝流程

通常YOLOv8的剪枝流程包含约束训练（稀疏训练）、剪枝、回调训练（微调）三个主要步骤：

1. **约束训练（稀疏训练）**：
   - **原理**：在BN层添加L1正则化，使得BN层的参数（γ, β）趋于0。当γ趋于0时，该通道上的卷积输出接近0，可认为该通道冗余，剔除对模型性能影响较小。
   - **代码实现**：在`./ultralytics/engine/trainer.py`中添加以下内容：

```python
l1_lambda = 1e - 2 * (1 - 0.9 * epoch / self.epochs)
for k, m in self.model.named_modules():    if isinstance(m, nn.BatchNorm2d):        m.weight.grad.data.add_(l1_lambda * torch.sign(m.weight.data))        m.bias.grad.data.add_(1e - 2 * torch.sign(m.bias.data))
```

```
- **启动训练**：
```

```python
from ultralytics import YOLO
model = YOLO('yolov8n.pt')
results = model.train(data='./data/data_nc5/data_nc5.yaml', batch=8, epochs=300, save=True, amp=False)
```

```
- **注意事项**：启动训练务必设置`amp=False`，关闭混合精度训练，否则无法得到结果。若稀疏训练时BN层参数过早全到0影响精度，可把系数`1e - 2`改小，如`1e - 3`。
```

2. **剪枝**：
- **选择剪枝模型**：选用上一步训练得到的模型，如`./runs/detect/train2/weights/last.pt`进行剪枝处理。
- **创建剪枝文件**：在`/yolov8/`下新建文件`prune.py`，示例代码如下：

```python
from ultralytics import YOLO
import torch
from ultralytics.nn.modules import Bottleneck, Conv, C2f, SPPF, Detect

yolo = YOLO("./runs/detect/train2/weights/last.pt")
model = yolo.model
ws = []
bs = []
for _, m in model.named_modules():    if isinstance(m, torch.nn.BatchNorm2d):        w = m.weight.abs().detach()        b = m.bias.abs().detach()        ws.append(w)        bs.append(b)

factor = 0.8  # 通道保留比率
ws = torch.cat(ws)
threshold = torch.sort(ws, descending=True)[0][int(len(ws) * factor)]

def _prune(c1, c2):    wet = c1.bn.weight.data.detach()    bis = c1.bn.bias.data.detach()    list = []    _threshold = threshold    while len(list) < 8:        list = torch.where(wet.abs() >= _threshold)[0]        _threshold = _threshold * 0.5    i = len(list)    c1.bn.weight.data = wet[list]    c1.bn.bias.data = bis[list]    c1.bn.running_var.data = c1.bn.running_var.data[list]    c1.bn.running_mean.data = c1.bn.running_mean.data[list]    c1.bn.num_features = i    c1.conv.weight.data = c1.conv.weight.data[list]    c1.conv.out_channels = i    if c1.conv.bias is not None:        c1.conv.bias.data = c1.conv.bias.data[list]    if not isinstance(c2, list):        c2 = [c2]    for item in c2:        if item is not None:            if isinstance(item, Conv):                conv = item.conv            else:                conv = item            conv.in_channels = i            conv.weight.data = conv.weight.data[:, list]

def prune(m1, m2):    if isinstance(m1, C2f):        m1 = m1.cv2    if not isinstance(m2, list):        m2 = [m2]    for i, item in enumerate(m2):        if isinstance(item, C2f) or isinstance(item, SPPF):            m2[i] = item.cv1    _prune(m1, m2)

for _, m in model.named_modules():    if isinstance(m, Bottleneck):        _prune(m.cv1, m.cv2)

for _, p in yolo.model.named_parameters():    p.requires_grad = True

# yolo.export(format="onnx")  # 导出为onnx文件
# yolo.train(data="VOC.yaml", epochs=100)  # 剪枝后直接训练微调
torch.save(yolo.ckpt, "./runs/detect/train2/weights/prune.pt")
```

```
- **注意事项**：由于NVIDIA的硬件加速原因，保留的通道数应大于等于8，可设置`local_threshold`尽量小，让更多通道保留下来。
```

3. **回调训练（微调）**：
- **代码修改**：
- 将先前在`./ultralytics/engine/trainer.py`中添加的L1正则化部分注释掉。
- 在该文件第543行左右添加代码`self.model = weights`。
- **再训练**：利用已经剪枝好的模型`prune.pt`，再次启动训练：

```python
from ultralytics import YOLO
model = YOLO("./runs/detect/train2/weights/prune.pt")
results = model.train(data='./data/data_nc5/data_nc5.yaml', batch=8, epochs=100, save=True)
```

### 其他剪枝方法及策略

1. **基于DepGraph（依赖图）的剪枝方法**：
   - **安装必要库**：

```bash
pip install torch torchvision ultralytics torch - pruner
```

```
- **加载预训练模型并构建依赖图**：
```

```python
from torch_pruner import DependencyGraph
from ultralytics import YOLO

model = YOLO('yolov8n.pt')
torch_model = model.model
dg = DependencyGraph()
dg.build_dependency(torch_model, example_inputs=torch.randn(1, 3, 640, 640))
```

```
- **剪枝策略设计**：
```

```python
# 设置全局剪枝比例
global_prune_ratio = 0.3  # 剪枝30%的通道
# 或者按层设置不同的剪枝比例
layer_prune_ratios = {
    'model.0.conv': 0.2,
    'model.1.conv': 0.3,
    # ...其他层配置
}
```

```
- **执行剪枝**：
```

```python
from torch_pruner import pruner

pruner = pruner.MagnitudePruner(
    model=torch_model,
    importance=channel_importance,
    global_prune_ratio=global_prune_ratio,
    dependency_graph=dg
)
pruner.prune()
print(torch_model)
```

```
- **模型微调**：
```

```python
# 定义微调参数
finetune_epochs = 50
learning_rate = 0.001
# 创建优化器
optimizer = torch.optim.Adam(torch_model.parameters(), lr=learning_rate)
# 微调循环
for epoch in range(finetune_epochs):    for images, targets in dataloader:        optimizer.zero_grad()        outputs = torch_model(images)        loss = compute_loss(outputs, targets)        loss.backward()        optimizer.step()        print(f'Epoch {epoch + 1}, Loss: {loss.item()}')
```

2. **分层剪枝策略**：根据层深度设置不同的剪枝比例，例如：

```python
def layer_depth_aware_prune_ratio(layer_name):    depth = int(layer_name.split('.')[1])  # 假设层名格式为model.x.conv    base_ratio = 0.3    # 深层网络剪枝比例较低    return base_ratio * (1 - 0.05 * depth)

# 应用分层剪枝
for layer in conv_layers:    ratio = layer_depth_aware_prune_ratio(layer.name)    pruner.set_layer_prune_ratio(layer, ratio)
```

3. **渐进式剪枝**：

```python
num_iterations = 5
total_prune_ratio = 0.5
for i in range(num_iterations):    current_ratio = (i + 1) / num_iterations * total_prune_ratio    pruner.global_prune_ratio = current_ratio    pruner.prune()    # 每次剪枝后微调    finetune_for_epochs(1)
```

### 剪枝后可能出现的问题及解决方法

1. **剪枝后模型崩溃**：
   - **问题现象**：剪枝后模型输出全为0或完全不合理。
   - **解决方法**：
     - 降低剪枝比例。
     - 确保剪枝后各层的通道数兼容。
     - 增加微调的epoch数。
     - 使用更小的学习率进行微调。
     - 采用渐进式剪枝策略。
     - 对敏感层使用更保守的剪枝比例。
     - 确保剪枝是结构化的。
     - 检查是否剪枝了计算密集型层。

### 总结

通过上述剪枝方法和策略，可以有效减少YOLOv8模型的参数量和计算量，提高模型的推理速度，同时通过微调等操作尽量保持模型的精度。在实际应用中，需要根据具体的任务需求和硬件资源，选择合适的剪枝方法和参数。
