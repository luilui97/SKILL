import os
import argparse
import time
from datetime import datetime
from utils import load_config, setup_logger
logger = setup_logger(process_name="prediction")
import json
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from data_loader import DataLoader


def cal_Y_prob(model:AutoModelForCausalLM, tokenizer:AutoTokenizer, generation_config, prompt_list):
    messages_batch = [
        [{"role": "user", "content": prompt}]
        for prompt in prompt_list
    ]
    texts = [tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) for messages in messages_batch]
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to("cuda")

    generated_output = model.generate(
        input_ids=inputs.input_ids,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        return_dict_in_generate=True,
        output_scores=True,
        **generation_config
    )
    
    scores = generated_output.scores[0]
    probs = scores.softmax(dim=-1)
    
    Y_id = tokenizer.encode("Y", add_special_tokens=False)[0]
    N_id = tokenizer.encode("N", add_special_tokens=False)[0]
    
    Y_probs = [probs[i, Y_id].item() for i in range(probs.shape[0])]
    
    return Y_probs


def log_results(label, results):
    logger.info(f"{label} Hits results: {results}\n")
    hit_at_1 = round(sum(1 for hits in results if hits == 1) / len(results), 3)
    mrr = round(sum(1 / hits for hits in results if hits != 0) / len(results), 3)
    logger.info(f"{label} Hit@1: {hit_at_1}\n")
    logger.info(f"{label} MRR: {mrr}\n")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default='configs/prediction/FB15k-237-subset-full.yaml',help="Path to the configuration file")
    args = parser.parse_args()
    logger.info(args.config)
    config = load_config(args.config)
    logger.info(config)
    dataset = config.get('dataset', 'FB15k-237-subset')
    setting = config.get('setting', 'transductive')
    train_size = config.get('train_size', 'full')
    max_path_hops = config.get('max_path_hops', 3)
    model_name = config.get('model','Qwen2-7B-Instruct')
    instruction_type = config.get('instruction_type', 'rule_path')
    data_loader = DataLoader(dataset=dataset, setting=setting, train_size=train_size, max_path_hops=max_path_hops)
    rule_dir = os.path.join(f"{data_loader.dataset_path}/rules_{data_loader.max_path_hops}_hop", f"filtered_rules_train_size_{data_loader.train_size}.json")
    with open(rule_dir, 'r', encoding='utf-8') as f:
        rules = json.load(f)


    test_batches = data_loader.get_test_batches()
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    generation_config = dict(
        temperature=0,
        top_k=0,
        top_p=0,
        do_sample=False,
        max_new_tokens=1,
    )

    llm_batch_size = 1
    sample_counter = 0
    hits_result_rule_path = []
    hits_result_rule_ne_path = []
    hits_result_average_ensemble = []
    hits_result_type = []
    logger.info(f"Using model: {model_name}\n")


    if instruction_type == 'rule_path':
        for idx, batch in enumerate(tqdm(test_batches, desc="Processing test batches")):
            rule_path_prompts = [data_loader.build_rule_path_prompt(test_triple, rules) for test_triple in batch]
            rule_path_probs = []
            for i in range(0, len(rule_path_prompts), llm_batch_size):
                batch_prompts = rule_path_prompts[i:i + llm_batch_size]
                rule_path_probs.extend(cal_Y_prob(model, tokenizer, generation_config, batch_prompts))

            rule_path_probs_in_batch = list(zip(rule_path_probs, range(len(rule_path_probs))))
            sorted_rule_path_indices = sorted(range(len(rule_path_probs_in_batch)), key=lambda i: rule_path_probs_in_batch[i][0], reverse=True)
            # logger.info(f"Sorted rule path indices: {sorted_rule_path_indices}\n")
            hits_position_base = sorted_rule_path_indices.index(0) + 1 if 0 in sorted_rule_path_indices else 0
            hits_result_rule_path.append(hits_position_base)
            # logger.info("*"*50 + "\n")

            if (idx + 1) % 400 == 0:
                logger.info(f"\nMetrics after processing {idx + 1} batches:\n")
                log_results("Rule Path", hits_result_rule_path)
                logger.info("\n" + "="*50 + "\n")
        
        logger.info("Final Results:\n")
        log_results("Rule Path", hits_result_rule_path)
        
    if instruction_type == 'rule_ne_path':
        for idx, batch in enumerate(tqdm(test_batches, desc="Processing test batches")):
            rule_ne_path_prompts = [data_loader.build_rule_ne_path_prompt(test_triple, rules) for test_triple in batch]
            rule_ne_path_probs = []
            for i in range(0, len(rule_ne_path_prompts), llm_batch_size):
                batch_prompts = rule_ne_path_prompts[i:i + llm_batch_size]
                rule_ne_path_probs.extend(cal_Y_prob(model, tokenizer, generation_config, batch_prompts))

            rule_ne_path_probs_in_batch = list(zip(rule_ne_path_probs, range(len(rule_ne_path_probs))))
            sorted_ne_rule_path_indices = sorted(range(len(rule_ne_path_probs_in_batch)), key=lambda i: rule_ne_path_probs_in_batch[i][0], reverse=True)
            # logger.info(f"Sorted rule path indices: {sorted_ne_rule_path_indices}\n")
            hits_position_base = sorted_ne_rule_path_indices.index(0) + 1 if 0 in sorted_ne_rule_path_indices else 0
            hits_result_rule_ne_path.append(hits_position_base)
            # logger.info("*"*50 + "\n")

            if (idx + 1) % 400 == 0:
                logger.info(f"\nMetrics after processing {idx + 1} batches:\n")
                log_results("Rule NE Path", hits_result_rule_ne_path)
                logger.info("\n" + "="*50 + "\n")
        
        logger.info("Final Results:\n")
        log_results("Rule NE Path", hits_result_rule_ne_path)



if __name__ == "__main__":
    main()
