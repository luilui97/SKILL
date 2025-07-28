import os
import json
from tqdm import tqdm
import argparse
from utils import load_config, setup_logger
logger = setup_logger(process_name="build_instruction")
from data_loader import DataLoader
from prompt_templates import RULE_PATH_PROMPT, RULE_NE_PATH_PROMPT

def build_instruction(data_loader:DataLoader, rules, neg_num=12, instruction_type='rule_path'):
    """
    Build instruction for the given triple and rules.
    """
    # rule_paths = rule_path_finder(data_loader, triple, rules)
    paths_dir = f"instructions/{data_loader.dataset}"
    os.makedirs(paths_dir, exist_ok=True)
    sft_instructions = []
    if instruction_type=='rule_path':
        for pos_triple in tqdm(data_loader.path_set, desc=f"Processing {data_loader.dataset} - setting: {data_loader.setting} - Train_size: {data_loader.train_size}"):
            logger.info(f"Processing positive triple: {pos_triple}")
            pos_head, relation, pos_tail = pos_triple
            removed_from_head = (relation, pos_tail, 1)
            removed_from_tail = (relation, pos_head, -1)
            data_loader.entity2relationtail_dict[pos_head].remove(removed_from_head)
            data_loader.entity2relationtail_dict[pos_tail].remove(removed_from_tail)

            pos_close_paths, pos_rule_candidates = data_loader.rule_path_finder(pos_triple, rules)
            pos_rule_paths_sentence = data_loader.rule_path_sentence(pos_rule_candidates, relation)
            pos_closed_paths_sentence = data_loader.closed_path_sentence(pos_close_paths)
            pos_head_entity_text = data_loader.entity2text.get(pos_head,"")
            pos_tail_entity_text = data_loader.entity2text.get(pos_tail,"")
           
            pos_instruction = RULE_PATH_PROMPT.format(
                head_entity=pos_head_entity_text,
                tail_entity=pos_tail_entity_text,
                reasoning_rules=pos_rule_paths_sentence,
                reasoning_paths=pos_closed_paths_sentence,
                test_triple=data_loader.triple_to_sentence(pos_triple)
            )
            sft_instructions.append({
                "instruction": pos_instruction,
                "input": "",
                "output": "Y"
            })

            neg_samples = data_loader.neg_sampling(pos_triple, neg_num)
            for neg_triple in neg_samples:
                neg_head, neg_relation, neg_tail = neg_triple
                neg_close_paths, neg_rule_candidates = data_loader.rule_path_finder(neg_triple, rules)
                neg_rule_paths_sentence = data_loader.rule_path_sentence(neg_rule_candidates, neg_relation)
                neg_closed_paths_sentence = data_loader.closed_path_sentence(neg_close_paths)
                neg_head_entity_text = data_loader.entity2text.get(neg_head, "")
                neg_tail_entity_text = data_loader.entity2text.get(neg_tail, "")
                neg_instruction = RULE_PATH_PROMPT.format(
                    head_entity=neg_head_entity_text,
                    tail_entity=neg_tail_entity_text,
                    reasoning_rules=neg_rule_paths_sentence,
                    reasoning_paths=neg_closed_paths_sentence,
                    test_triple= data_loader.triple_to_sentence(neg_triple)
                )
                sft_instructions.append({
                    "instruction": neg_instruction,
                    "input": "",
                    "output": "N"
                })

            data_loader.entity2relationtail_dict[pos_head].append(removed_from_head)
            data_loader.entity2relationtail_dict[pos_tail].append(removed_from_tail)

    elif instruction_type=='rule_ne_path':
        for pos_triple in tqdm(data_loader.path_set, desc=f"Processing {data_loader.dataset} - setting: {data_loader.setting} - Train_size: {data_loader.train_size}"):
            logger.info(f"Processing positive triple: {pos_triple}")
            pos_head, relation, pos_tail = pos_triple
            removed_from_head = (relation, pos_tail, 1)
            removed_from_tail = (relation, pos_head, -1)
            data_loader.entity2relationtail_dict[pos_head].remove(removed_from_head)
            data_loader.entity2relationtail_dict[pos_tail].remove(removed_from_tail)

            pos_neighbor_triples = data_loader.neighbor_triple_finder_rule(pos_triple,rules)

            pos_close_paths, pos_rule_candidates = data_loader.rule_path_finder(pos_triple, rules)
            pos_rule_paths_sentence = data_loader.rule_path_sentence(pos_rule_candidates, relation)
            pos_closed_paths_sentence = data_loader.closed_path_sentence(pos_close_paths)
            pos_head_entity_text = data_loader.entity2text.get(pos_head,"")
            pos_tail_entity_text = data_loader.entity2text.get(pos_tail,"")
            # pos_tail_entity_text = data_loader.entity2description.get(pos_tail, data_loader.entity2text[pos_tail])
            pos_instruction = RULE_NE_PATH_PROMPT.format(
                head_entity=pos_head_entity_text,
                tail_entity=pos_tail_entity_text,
                neighbor_triples='\n'.join(pos_neighbor_triples),
                reasoning_rules=pos_rule_paths_sentence,
                reasoning_paths=pos_closed_paths_sentence,
                test_triple=data_loader.triple_to_sentence(pos_triple)
            )
            sft_instructions.append({
                "instruction": pos_instruction,
                "input": "",
                "output": "Y"
            })

            neg_samples = data_loader.neg_sampling(pos_triple, neg_num)
            for neg_triple in neg_samples:
                neg_head, neg_relation, neg_tail = neg_triple


                neg_neighbor_triples = data_loader.neighbor_triple_finder_rule(neg_triple,rules)
                
                neg_close_paths, neg_rule_candidates = data_loader.rule_path_finder(neg_triple, rules)
                neg_rule_paths_sentence = data_loader.rule_path_sentence(neg_rule_candidates, neg_relation)
                neg_closed_paths_sentence = data_loader.closed_path_sentence(neg_close_paths)
                neg_head_entity_text = data_loader.entity2text.get(neg_head, "")
                neg_tail_entity_text = data_loader.entity2text.get(neg_tail, "")
                neg_instruction = RULE_NE_PATH_PROMPT.format(
                    head_entity=neg_head_entity_text,
                    tail_entity=neg_tail_entity_text,
                    neighbor_triples='\n'.join(neg_neighbor_triples),
                    reasoning_rules=neg_rule_paths_sentence,
                    reasoning_paths=neg_closed_paths_sentence,
                    test_triple= data_loader.triple_to_sentence(neg_triple)
                )
                sft_instructions.append({
                    "instruction": neg_instruction,
                    "input": "",
                    "output": "N"
                })

            data_loader.entity2relationtail_dict[pos_head].append(removed_from_head)
            data_loader.entity2relationtail_dict[pos_tail].append(removed_from_tail)

    
    

    sft_instructions_path = f"{paths_dir}/{data_loader.dataset}_train_size_{data_loader.train_size}_{instruction_type}_rule.json"
    with open(sft_instructions_path, "w", encoding="utf-8") as f:
        json.dump(sft_instructions, f, ensure_ascii=False, indent=4)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default='configs/build_instruction/FB15k-237-subset-full_rule_path.yaml',help="Path to the configuration file")
    args = parser.parse_args()
    logger.info(args.config)
    config = load_config(args.config)
    logger.info(config)
    dataset = config.get('dataset', 'FB15k-237-subset')
    setting = config.get('setting', 'transductive')
    train_size = config.get('train_size', 'full')
    max_path_hops = config.get('max_path_hops', 3)
    instruction_type = config.get('instruction_type', 'rule_path')
    data_loader = DataLoader(dataset=dataset, setting=setting, train_size=train_size, max_path_hops=max_path_hops)
    rule_dir = os.path.join(f"{data_loader.dataset_path}/rules_{data_loader.max_path_hops}_hop", f"filtered_rules_train_size_{data_loader.train_size}.json")
    with open(rule_dir, 'r', encoding='utf-8') as f:
        rules = json.load(f)
    build_instruction(data_loader, rules, neg_num=12, instruction_type=instruction_type)
    
if __name__ == "__main__":
    main()
