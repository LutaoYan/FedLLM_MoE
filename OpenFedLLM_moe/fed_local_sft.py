import torch
import copy
import sys
import os
sys.path.append('/home/wupanlong/research/OpenFedLLM_moe/federated_learning')
from trl import SFTTrainer
import warnings
from .my_sfttrainer import MySFTTrainer
from transformers import TrainerCallback
from transformers.utils.import_utils import is_sagemaker_mp_enabled, is_apex_available
from transformers.trainer_utils import set_seed, enable_full_determinism, get_last_checkpoint,find_executable_batch_size
from transformers.trainer_callback import TrainerState
# from transformers.trainer_pt_utils import smp_forward_backward
from peft import get_peft_model_state_dict, set_peft_model_state_dict
from sentence_transformers import SentenceTransformer
import numpy as np
from torch import nn
import torch.nn.functional as F
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
TRAINER_STATE_NAME = "trainer_state.json"
if TYPE_CHECKING:
    import optuna
if is_apex_available():
    from apex import amp

def get_fed_local_sft_trainer_withkd_moe(script_args, fed_args, model, tokenizer, training_args, local_dataset, formatting_prompts_func, data_collator, global_dict, local_auxiliary, global_auxiliary):
    # train with KD and moe
    if (fed_args.fed_alg in ['fedavg']) or (fed_args.fed_alg).startswith('local'):
        trainer = SFTTrain_withkd(
            model=model,
            tokenizer=tokenizer,
            args=training_args,
            max_seq_length=script_args.seq_length,
            train_dataset=local_dataset,
            formatting_func=formatting_prompts_func,
            data_collator=data_collator,
        )
    else:
        raise ValueError(f'Unsupported `fed_alg`: {fed_args.fed_alg}')
    return trainer

def get_fed_local_sft_trainer_withmoe(script_args, fed_args, model, tokenizer, training_args, local_dataset, formatting_prompts_func, data_collator, global_dict, local_auxiliary, global_auxiliary):
    # train with  moe
    if (fed_args.fed_alg in ['fedavg']) or (fed_args.fed_alg).startswith('local'):
        trainer = SFTTrain_withkd(
            model=model,
            tokenizer=tokenizer,
            args=training_args,
            max_seq_length=script_args.seq_length,
            train_dataset=local_dataset,
            formatting_func=formatting_prompts_func,
            data_collator=data_collator,
        )
    else:
        raise ValueError(f'Unsupported `fed_alg`: {fed_args.fed_alg}')
    return trainer

def get_fed_local_sft_trainer_withkd(script_args, fed_args, model, tokenizer, training_args, local_dataset, formatting_prompts_func, data_collator, global_dict, local_auxiliary, global_auxiliary, logits_distill=False):
    # train with KD
    if (fed_args.fed_alg in ['fedavg']) or (fed_args.fed_alg).startswith('local'):
        trainer = SFTTrain_withkd(
            model=model,
            tokenizer=tokenizer,
            args=training_args,
            max_seq_length=script_args.seq_length,
            train_dataset=local_dataset,
            formatting_func=formatting_prompts_func,
            data_collator=data_collator,
            logits_distill=logits_distill,
            # packing=True,
        )
    else:
        raise ValueError(f'Unsupported `fed_alg`: {fed_args.fed_alg}')
    return trainer


def get_fed_local_sft_trainer(script_args, fed_args, model, tokenizer, training_args, local_dataset,eval_dataset, formatting_prompts_func, data_collator, global_dict, local_auxiliary, global_auxiliary):
    
    if fed_args.fed_alg == 'fedprox':
        trainer = SFTTrainerFedProx(
            model=model,
            tokenizer=tokenizer,
            args=training_args,
            max_seq_length=script_args.seq_length,
            train_dataset=local_dataset,
            formatting_func=formatting_prompts_func,
            data_collator=data_collator,
            global_state=global_dict,
            prox_mu=fed_args.prox_mu,
        )
    elif fed_args.fed_alg == 'scaffold':
        trainer = SFTTrainerSCAFFOLD(
            model=model,
            tokenizer=tokenizer,
            args=training_args,
            max_seq_length=script_args.seq_length,
            train_dataset=local_dataset,
            formatting_func=formatting_prompts_func,
            data_collator=data_collator,
            global_state=global_dict,
            local_auxiliary=local_auxiliary,
            global_auxiliary=global_auxiliary,
        )
        trainer.add_callback(SCAFFOLD_Callback(trainer.correction, model))
    elif (fed_args.fed_alg in ['fedavg']) or (fed_args.fed_alg).startswith('local'):
        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            args=training_args,
            max_seq_length=script_args.seq_length,
            train_dataset=local_dataset,
            eval_dataset = eval_dataset,
            formatting_func=formatting_prompts_func,
            data_collator=data_collator,
        )
    else:
        raise ValueError(f'Unsupported `fed_alg`: {fed_args.fed_alg}')
    return trainer

class SFTTrainerFedProx(SFTTrainer):
    def __init__(self, global_state, prox_mu, **kwargs):
        super(SFTTrainerFedProx, self).__init__(**kwargs)
        self.global_state = global_state
        self.mu = prox_mu
    
    def compute_loss(self, model, inputs, return_outputs=False):

        return_values = super(SFTTrainerFedProx, self).compute_loss(model, inputs, return_outputs=return_outputs)

        if return_outputs:
            loss, outputs = return_values
        else:
            loss = return_values

        # Apply FedProx Loss
        for name, param in model.named_parameters():
            name = name.replace(".default", "")     # TODO: May need changes. to accord with peft
            # only trainable parameters
            if not param.requires_grad:
                continue
            else:
                loss += self.mu / 2 * torch.norm(param - self.global_state[name]) ** 2

        return (loss, outputs) if return_outputs else loss


class SFTTrainerSCAFFOLD(SFTTrainer):
    def __init__(self, global_state, local_auxiliary, global_auxiliary, **kwargs):
        super(SFTTrainerSCAFFOLD, self).__init__(**kwargs)
        self.global_state = global_state
        self.local_auxiliary = local_auxiliary
        self.global_auxiliary = global_auxiliary
        self.correction = copy.deepcopy(local_auxiliary)

        for name in self.correction.keys():
            self.correction[name] = self.global_auxiliary[name] - self.local_auxiliary[name]
    
    def get_auxiliary_param(self):
        auxiliary_new_para = copy.deepcopy(self.local_auxiliary)
        auxiliary_delta_para = copy.deepcopy(self.local_auxiliary)
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if not param.requires_grad:
                    continue
                else:
                    name = name.replace(".default", "")
                    auxiliary_new_para[name] = (self.global_state[name] - param) / (self.args.max_steps * self.args.learning_rate) - self.correction[name]
                    auxiliary_delta_para[name] = auxiliary_new_para[name] - self.local_auxiliary[name]
        return auxiliary_new_para, auxiliary_delta_para


class SFTTrain_withkd(MySFTTrainer):
    def __init__(self, logits_distill=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # modified
        self.logits_distill = logits_distill
    
    # def _inner_training_loop(self, batch_size=None, args=None, resume_from_checkpoint=None, trial=None, ignore_keys_for_eval=None):
    #     return super()._inner_training_loop(batch_size, args, resume_from_checkpoint, trial, ignore_keys_for_eval)
    
    def train(
        self,
        resume_from_checkpoint: Optional[Union[str, bool]] = None,
        trial: Union["optuna.Trial", Dict[str, Any]] = None,
        ignore_keys_for_eval: Optional[List[str]] = None,
        **kwargs,
    ):
        if resume_from_checkpoint is False:
            resume_from_checkpoint = None

        # memory metrics - must set up as early as possible
        self._memory_tracker.start()
        

        args = self.args

        self.is_in_train = True

        # Attach NEFTune hooks if necessary
        if self.neftune_noise_alpha is not None:
            self.model = self._activate_neftune(self.model)

        # do_train is not a reliable argument, as it might not be set and .train() still called, so
        # the following is a workaround:
        if (args.fp16_full_eval or args.bf16_full_eval) and not args.do_train:
            self._move_model_to_device(self.model, args.device)

        if "model_path" in kwargs:
            resume_from_checkpoint = kwargs.pop("model_path")
            warnings.warn(
                "`model_path` is deprecated and will be removed in a future version. Use `resume_from_checkpoint` "
                "instead.",
                FutureWarning,
            )
        if len(kwargs) > 0:
            raise TypeError(f"train() received got unexpected keyword arguments: {', '.join(list(kwargs.keys()))}.")
        # This might change the seed so needs to run first.
        self._hp_search_setup(trial)
        self._train_batch_size = self.args.train_batch_size

        # Model re-init
        model_reloaded = False
        if self.model_init is not None:
            # Seed must be set before instantiating the model when using model_init.
            enable_full_determinism(self.args.seed) if self.args.full_determinism else set_seed(self.args.seed)
            self.model = self.call_model_init(trial)
            model_reloaded = True
            # Reinitializes optimizer and scheduler
            self.optimizer, self.lr_scheduler = None, None

        # Load potential model checkpoint
        if isinstance(resume_from_checkpoint, bool) and resume_from_checkpoint:
            resume_from_checkpoint = get_last_checkpoint(args.output_dir)
            if resume_from_checkpoint is None:
                raise ValueError(f"No valid checkpoint found in output directory ({args.output_dir})")

        if (
            resume_from_checkpoint is not None
            and not is_sagemaker_mp_enabled()
            and not self.is_deepspeed_enabled
            and not self.is_fsdp_enabled
        ):
            self._load_from_checkpoint(resume_from_checkpoint)
            # In case of repeating the find_executable_batch_size, set `self._train_batch_size` properly
            state = TrainerState.load_from_json(os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME))
            if state.train_batch_size is not None:
                self._train_batch_size = state.train_batch_size

        # If model was re-initialized, put it on the right device and update self.model_wrapped
        if model_reloaded:
            if self.place_model_on_device:
                self._move_model_to_device(self.model, args.device)
            self.model_wrapped = self.model

        inner_training_loop = find_executable_batch_size(
            self._inner_training_loop, self._train_batch_size, args.auto_find_batch_size
        )
    
        return inner_training_loop(
            args=args,
            resume_from_checkpoint=resume_from_checkpoint,
            trial=trial,
            ignore_keys_for_eval=ignore_keys_for_eval,
        )
    
    #! modified 
    def compute_loss(self, model, past_model, inputs, return_outputs=False):
        # origin  cross entropy loss
        ce_loss = super(SFTTrain_withkd, self).compute_loss(model, inputs, return_outputs=return_outputs)
        if self.logits_distill == False:
            ## KD throuth LLM capability
            # 加载预训练的SBERT模型，这里以'sbert-base-nli-mean-tokens'为例
            similarity_model = SentenceTransformer('/home/wupanlong/research/OpenFedLLM_moe/weights/minilm')
            # print((inputs.data['input_ids']).shape[1])
            # print(inputs.data['attention_mask'])
            pure_input = inputs.data['input_ids'][inputs.data['labels'] == -100].unsqueeze(0)
            # processed_input = self.tokenizer.decode(pure_input, skip_special_tokens=True)
            # my_input = [self.tokenizer.decode(output_str, skip_special_tokens=True) for output_str in inputs.data['input_ids']]
            # print(my_input)
            
            curoutputs_str = model.generate(input_ids=pure_input, max_new_tokens=1024, repetition_penalty=1.1)
            # curoutputs_str = model.generate(input_ids=processed_input, max_new_tokens=512)[:, (processed_input).shape[1]:]
            pastoutputs_str = past_model.generate(input_ids=pure_input, max_new_tokens=1024, repetition_penalty=1.1)
            past_result = [self.tokenizer.decode(output_str, skip_special_tokens=True) for output_str in pastoutputs_str]
            cur_result = [self.tokenizer.decode(output_str, skip_special_tokens=True) for output_str in curoutputs_str]
            print(past_result)
            print(cur_result)
            
            
            # 计算句子嵌入向量
            cosine_similarity = 0
            for i in range(len(past_result)):
                embeddings1 = similarity_model.encode(past_result[i])
                embeddings2 = similarity_model.encode(cur_result[i])
 
                # 使用余弦相似度计算两个句子向量之间的相似度
                cosine_similarity += np.dot(embeddings1, embeddings2.T)
                print('the value of cosine_similarity is', cosine_similarity)
            kd_loss = 1 - cosine_similarity 

            total_loss = ce_loss + kd_loss

            del pure_input, pastoutputs_str, curoutputs_str, embeddings1, embeddings2, similarity_model
            torch.cuda.empty_cache()
        
        if self.logits_distill == True:
            # compute kd loss of the past model and the current model by using the KL divergence of the output distribution
            past_outputs = past_model(**inputs)
            cur_outputs = model(**inputs)

            pastoutput_logits = past_outputs['logits']
            curoutput_logits = cur_outputs['logits']
            # predicted_token_ids = curoutput_logits.argmax(dim=-1)
            # 将token IDs转换为文本
            # 注意：这里我们只处理batch中的第一个样本，实际使用时你可能需要遍历batch
            # predicted_text = self.tokenizer.decode(predicted_token_ids[0], skip_special_tokens=True)
            # print(predicted_text)
            # encode_result = [self.tokenizer.decode(output_str, skip_special_tokens=True) for output_str in curoutput_logits]
            
            norm_pastoutput_logits = torch.nn.functional.softmax(pastoutput_logits, dim=-1)
            norm_curoutput_logits = torch.nn.functional.softmax(curoutput_logits, dim=-1)
            
            kl_divergence = F.kl_div(norm_pastoutput_logits.log(), norm_curoutput_logits, reduction='sum')
            # print('ce loss is', ce_loss)
            # print('kd loss is', kl_divergence)
            total_loss = ce_loss + kl_divergence / 50

        return total_loss
    
    #! modified 
    def training_step(self, model: nn.Module, past_model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        for name, param in model.named_parameters():
            if 'lora' in name and '10' in name and 'self_attn.v_proj.lora_B' in name:
                print(f"Name: {name}, Value: {param}")
        
        for name, param in past_model.named_parameters():
            if 'lora' in name and '10' in name and 'self_attn.v_proj.lora_B' in name:
                print(f"Name: {name}, Value: {param}")
        
        
        model.train()
        inputs = self._prepare_inputs(inputs)

        # if is_sagemaker_mp_enabled():
        #     loss_mb = smp_forward_backward(model, inputs, self.args.gradient_accumulation_steps)
        #     return loss_mb.reduce_mean().detach().to(self.args.device)
        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, past_model, inputs)

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        if self.use_apex:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            self.accelerator.backward(loss)

        return loss.detach() / self.args.gradient_accumulation_steps


class SCAFFOLD_Callback(TrainerCallback):
    def __init__(self, correction, model):
        super(SCAFFOLD_Callback, self).__init__()
        self.correction = correction
        self.model = model
    def on_step_end(self, args, state, control, **kwargs):
        model_para = copy.deepcopy(get_peft_model_state_dict(self.model))
        for name in model_para.keys():
            model_para[name] -= args.learning_rate * self.correction[name]
        set_peft_model_state_dict(self.model, model_para)
        