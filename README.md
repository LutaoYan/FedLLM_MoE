My research about Federated Learning & LoRA & MoE & LLM.


# 修改内容 
## 添加文件
FED/OpenFedLLM_moe/utils/ylt_module.py
## 文件moe_traing.py
记得import ylt_module
### 启用aux_loss
line 48 obalance = True
### 数据集划分
line 70-80
### 模型验证支持
FED/OpenFedLLM_moe/federated_learning/fed_local_sft.py  line 112 在avg算法函数加入 eval_dataset = eval_dataset,
line 154-156  line 158
line 241--- 实例化一个model的trainer然后进行验证集验证
### 数据保存
line277--- 保存每一轮训练的loss，还有每一轮训练最后一批验证集的Token分布，保存到两个文件中

## 文件 FED/OpenFedLLM_moe/src/mola_modeling_llama_hacked.py
### aux_loss系数
### 修改函数
### 修改FFN返回


