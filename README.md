My research about Federated Learning & LoRA & MoE & LLM.


# 修改内容
- 修改的内容都放在了OpenFedLLM_moe文件夹下
- 主要修改地方也会有"#ylt_modified标识
## 添加文件
FED/OpenFedLLM_moe/utils/ylt_module.py
- 里面封装了一些函数 都是在moe_training.py里才会用到
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
line277--- 保存每一轮训练的loss，还有每一轮训练最后一批验证集的Token分布，保存到两个文件中 被存到output文件夹

## 文件 FED/OpenFedLLM_moe/src/mola_modeling_llama_hacked.py
### aux_loss系数
line 1149 self.router_aux_loss_coef
### 修改函数
line 21-75 原函数被我注释掉
line 77 load_balancing_loss()返回两个数据 一个是aux_loss,一个是token在expert的分配情况 对应修改line 1320
### 修改FFN返回
line 1158 新增eval_records = [] 记录每次经过FFN时不同层token的expert负载

line 1332-1340

## analysis文件夹
主要是存放了分析与可视化在一次代码运行完后的validation与training_loss 与热力图负载 用到output里面的文件 可自由修改



