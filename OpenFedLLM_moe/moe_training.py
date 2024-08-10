import copy
import os
import json
from tqdm import tqdm
import numpy as np
from peft import PeftModel
from trl import SFTTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DataCollatorForCompletionOnlyLM
from peft import get_peft_model_state_dict, set_peft_model_state_dict, prepare_model_for_kbit_training
from src.mola_mapping_hacked import get_peft_model
from federated_learning.fed_local_sft import get_fed_local_sft_trainer_withkd, get_fed_local_sft_trainer_withkd_moe
from utils import *
from federated_learning import *
from config import get_config, save_config, get_model_config, get_training_args
# from src.mola_trainer_hacked import Trainer
from src.mola_mapping_hacked import get_peft_model
from src.mola_lora_hacked import LoraConfig
from src.mola_peft_model_hacked import set_peft_model_state_dict_moe

from transformers import LlamaTokenizer, AutoConfig
from src.mola_modeling_llama_hacked import LlamaForCausalLM_d
from utils.ylt_module import *
# from modelscope import snapshot_download

# ===== Define the arguments =====
# lora hyperparams
# llama 2
lora_r = "8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8"
lora_alpha = 16
lora_dropout = 0.05
lora_target_modules = "q_proj,v_proj,k_proj,o_proj,gate_proj,down_proj,up_proj"
# mola hyperparams
# number_experts = "2,2,2,2,2,2,2,2,4,4,4,4,4,4,4,4,6,6,6,6,6,6,6,6,8,8,8,8,8,8,8,8"
# number_experts = "5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5"
number_experts = "6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6"

top_k = "2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2"
# tinyllama
# lora_r = "8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8"
# lora_alpha = 16
# lora_dropout = 0.05
# lora_target_modules = "q_proj,v_proj,k_proj,o_proj,gate_proj,down_proj,up_proj"
# # mola hyperparams
# number_experts = "2,2,2,2,2,2,2,2,4,4,4,4,4,4,4,4,6,6,6,6,6,6,6,6"
# top_k = "2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2"

obalance = True

lora_r = lora_r.split(",")
number_experts = number_experts.split(",")
top_k = top_k.split(",")
lora_target_modules = lora_target_modules.split(",")

lora_r = [int(lr) for lr in lora_r]
number_experts = [int(lr) for lr in number_experts]
top_k = [int(lr) for lr in top_k]
lora_target_modules = [str(lr) for lr in lora_target_modules]


script_args, fed_args, peft_config = get_config()
training_args = get_training_args(script_args, script_args.learning_rate)
save_config(script_args, fed_args)
print(script_args, fed_args)


# ===== Load the dataset =====
dataset = get_dataset(script_args.dataset_name, script_args.local_data_dir)
dataset = process_sft_dataset(script_args.dataset_name, dataset, script_args.dataset_sample)

#ylt_modified
#==== Spilt the dataset======
train_dataset_full, val_dataset_full, test_dataset = split_dataset_for_validation(dataset)
#取训练集1/2
tra_sample_ratio =0.5
tra_sample_size = int(tra_sample_ratio * len(train_dataset_full))
train_dataset = train_dataset_full.select(range(tra_sample_size))
# 取验证集1/20
val_sample_ratio = 0.05
val_sample_size = int(val_sample_ratio * len(val_dataset_full))
val_dataset = val_dataset_full.select(range(val_sample_size))
# print(dataset[0])

# ===== Split the dataset into clients =====
local_datasets = split_dataset(fed_args, script_args, train_dataset)
sample_num_list = [len(local_datasets[i]) for i in range(fed_args.num_clients)]

# ===== Get model config =====
device_map, quantization_config, torch_dtype = get_model_config(script_args)

# model = AutoModelForCausalLM.from_pretrained(
#     script_args.model_name_or_path,
#     quantization_config=quantization_config,
#     device_map=device_map,
#     trust_remote_code=script_args.trust_remote_code,
#     torch_dtype=torch_dtype,
# )

# modified
config = AutoConfig.from_pretrained(script_args.model_name_or_path)
config.lora_target_modules = lora_target_modules
device_map = 'auto'
model = LlamaForCausalLM_d.from_pretrained(
        script_args.model_name_or_path,
        quantization_config=quantization_config,
        config=config,
        # load_in_8bit=False,
        load_in_4bit=script_args.load_in_4bit,
        torch_dtype=torch_dtype,
        device_map=device_map,
)
# modified
print(model)
model.get_new_parameters(number_experts, top_k, obalance)

if script_args.from_scrach == False:
    model = PeftModel.from_pretrained(model, script_args.past_model_path, torch_dtype=torch_dtype)

if script_args.load_in_8bit or script_args.load_in_4bit:
    model = prepare_model_for_kbit_training(
                model, use_gradient_checkpointing=training_args.gradient_checkpointing
            )

config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=lora_target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
)
model = get_peft_model(model, config, number_experts=number_experts, top_k=top_k)
model.print_trainable_parameters()

# ===== Define the global and local models =====
global_dict = copy.deepcopy(get_peft_model_state_dict(model))
local_dict_list = [copy.deepcopy(global_dict) for i in range(fed_args.num_clients)]
proxy_dict, opt_proxy_dict = get_proxy_dict(fed_args, global_dict)
global_auxiliary, auxiliary_model_list, auxiliary_delta_dict = get_auxiliary_dict(fed_args, global_dict)

# ===== Define the tokenizer =====
tokenizer = AutoTokenizer.from_pretrained(script_args.model_name_or_path, use_fast=False, padding_side="right")
if tokenizer.pad_token is None:
    # tokenizer.pad_token = tokenizer.unk_token   # following vicuna
    tokenizer.pad_token = tokenizer.eos_token

# ===== Define the formatting function (cater to TRL SFTTrainer)=====
formatting_prompts_func, response_template = get_formatting_prompts_func(script_args.template, tokenizer.eos_token)
response_template_ids = tokenizer.encode(response_template, add_special_tokens=False)[2:]   # Now we have it like in the dataset texts: `[2277, 29937, 4007, 22137, 29901]` for Llama2
data_collator = DataCollatorForCompletionOnlyLM(response_template_ids, tokenizer=tokenizer)

# ===== Start federated training =====
training_loss = [[] for i in range(fed_args.num_clients)]
# ylt_modified
validation_loss = []  # 新增全局验证损失列表
eval_metrix = []

for round in tqdm(range(fed_args.num_rounds)):

    clients_this_round = get_clients_this_round(fed_args, round)

    print(f">> ==================== Round {round+1} : {clients_this_round} ====================")
    
    for client in range(fed_args.num_clients):

        if client not in clients_this_round:
            training_loss[client].append(-1)            # -1 is an indicator of not training
            continue

        set_peft_model_state_dict(model, global_dict)   # sync the global model to the local model

        sub_dataset = get_dataset_this_round(local_datasets[client], round, fed_args, script_args)      # get the required sub-dataset for this round
        new_lr = cosine_learning_rate(round, fed_args.num_rounds, script_args.learning_rate, 1e-6)      # manually schedule the learning rate
        training_args = get_training_args(script_args, new_lr)

        # ===== Train local model on the client side =====
        if script_args.from_scrach == True:
            trainer = get_fed_local_sft_trainer(
                model=model,
                tokenizer=tokenizer,
                training_args=training_args,
                local_dataset=sub_dataset,
                eval_dataset = val_dataset,
                formatting_prompts_func=formatting_prompts_func,
                data_collator=data_collator,
                global_dict=global_dict,
                fed_args=fed_args,
                script_args=script_args,
                local_auxiliary=auxiliary_model_list[client],
                global_auxiliary=global_auxiliary,
            )
        if script_args.from_scrach == False:
            # only with kd
            trainer = get_fed_local_sft_trainer_withkd(
                model=model,
                tokenizer=tokenizer,
                training_args=training_args,
                local_dataset=sub_dataset,
                formatting_prompts_func=formatting_prompts_func,
                data_collator=data_collator,
                global_dict=global_dict,
                fed_args=fed_args,
                script_args=script_args,
                local_auxiliary=auxiliary_model_list[client],
                global_auxiliary=global_auxiliary,
            )
            
            # with kd and moe
            trainer = get_fed_local_sft_trainer_withkd_moe(
                model=model,
                tokenizer=tokenizer,
                training_args=training_args,
                local_dataset=sub_dataset,
                formatting_prompts_func=formatting_prompts_func,
                data_collator=data_collator,
                global_dict=global_dict,
                fed_args=fed_args,
                script_args=script_args,
                local_auxiliary=auxiliary_model_list[client],
                global_auxiliary=global_auxiliary,
            )
            
            
        results = trainer.train()
        training_loss[client].append(results.training_loss)

        # ===== Client transmits local information to server =====
        if fed_args.fed_alg == 'scaffold':
            auxiliary_model_list[client], auxiliary_delta_dict[client] = trainer.get_auxiliary_param()

        local_dict_list[client] = copy.deepcopy(get_peft_model_state_dict(model))   # copy is needed!

    # ===== Server aggregates the local models =====
    global_dict, global_auxiliary = global_aggregate(
        fed_args, global_dict, local_dict_list, sample_num_list, \
        clients_this_round, round, proxy_dict=proxy_dict, \
        opt_proxy_dict=opt_proxy_dict, auxiliary_info=(global_auxiliary, auxiliary_delta_dict)
    )
    set_peft_model_state_dict(model, global_dict)   # Update global modelz

# ylt_modified    
    # ===== Calculate validation loss =====
    # # 创建模型的深拷贝，用于加速验证  undo
    # val_model = copy.deepcopy(model)
    # # 确保验证模型使用FP16
    # val_model.half()
    # # 创建验证用的TrainingArguments实例，并启用FP16
    # val_training_args = copy.deepcopy(training_args)
    # val_training_args.fp16 = True
    global_trainer =  SFTTrainer(
                model=model,
                tokenizer=tokenizer,
                args=training_args,
                train_dataset=sub_dataset,
                eval_dataset=val_dataset,
                formatting_func=formatting_prompts_func,
                data_collator=data_collator,
            )
    # Evaluate the model on the validation dataset
    eval_results = global_trainer.evaluate()
    # Extract the validation loss from the evaluation results
    val_loss = eval_results["eval_loss"]
    validation_loss.append(val_loss)
    #eval_metrix第一层数组代表每一轮evaluate(对应num_rounds50轮),第二层数组为val_dataset每次经过modelFFN记录下来的expert_statistics,每次sample最后一批token进行分析
    eval_metrix.append(global_trainer.model.eval_records[-1])
    eval_metrix_serializable = convert_tensor_to_serializable(eval_metrix)
    print("Contains Tensor:", check_for_tensors(eval_metrix_serializable))
    print(f"Validation Loss after round {round+1}: {val_loss}")


    # ===== Save the model =====
    # if (round+1) % 50 == 0:
    if round + 1 == 1:
        trainer.save_model(os.path.join(script_args.output_dir, script_args.dataset_name))
        print('the model saved at' + str(os.path.join(script_args.output_dir, script_args.dataset_name)))
    
    np.save(os.path.join(script_args.output_dir, "training_loss.npy"), np.array(training_loss))
    np.save(os.path.join(script_args.output_dir, "validation_loss.npy"), np.array(validation_loss))
    # Save evaluation metrics to a JSON file in the output directory
    json_output_path = os.path.join(script_args.output_dir, 'eval_metrics.json')
    with open(json_output_path, 'w', encoding='utf-8') as f:
        json.dump(eval_metrix_serializable, f, ensure_ascii=False, indent=4)
