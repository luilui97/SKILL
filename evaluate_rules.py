import ollama
import json
from utils import setup_logger, load_config
logger = setup_logger(process_name="evaluate_rules")
from data_loader import DataLoader
from tqdm import tqdm
from collections import Counter
import yaml
import argparse
import os


template = """
You are an expert in knowledge reasoning and rule-based inference. Your task is to evaluate the following reasoning rule and its instance.

Your evaluation should consider two aspects:

1. Reasonableness:

Does the rule logically follow from known facts or principles?

Are the premises valid and do they logically support the conclusion?

Is there sufficient evidence to justify this inference?

Is the rule premise the same? If so, the rule is not reasonable.

2. Usefulness:

Can this rule be applied in practical real-world scenarios?

Can it contribute to meaningful inference or prediction?

Does it help to reduce uncertainty, assist decision making, or generate new knowledge?

Decision Criteria:
If the rule is both reasonable and useful, answer "Yes".
If the rule fails to meet either Reasonableness or Usefulness, answer "No".

Here is an example evaluation:
Rule Head: person gender
Rule Premise:
person spouse s marriage type of union.
person spouse s marriage type of union.
person gender
Explanation: This rule tries to infer a person's gender based on having a spouse and knowing the marriage type of union. However, knowing someone is married and the marriage type does not allow inference of gender. Therefore, this rule is not reasonable or useful.
Answer: No

Now evaluate the following rule:

Rule Head:
{}

Rule Premise:
{}

Instance:
Result: {}
Premises: {}

Please answer only with "Yes" or "No". Do not provide any additional explanation or context.

"""

def evaluate_rule(data_loader, model_name="llama3.1:70b"):
    filtered_rules = {}
    # save_path = "filtered_rules_70b.json"
    save_path = os.path.join(f"{data_loader.dataset_path}/rules_{data_loader.max_path_hops}_hop", f"filtered_rules_train_size_{data_loader.train_size}.json")
    # save_frequency = 100
    for key,values in tqdm(data_loader.raw_rules.items()):
        target_relation = values.get("relation","")
        primise = values.get("relation_premise","")
        primise_sentence = '.\n'.join([data_loader.relation2text[p] for p in primise])
        path_instance = values.get("path_instances",[])
        confidence = values.get("confidence", 0.0)
        if path_instance:
            target_triple = data_loader.triple_to_sentence(tuple(path_instance[0][0]))
            promise_triple = '\n'.join([data_loader.triple_to_sentence(tuple(t)) for t in path_instance[0][1:]])
    
        prompt = template.format(
            data_loader.relation2text[target_relation],
            primise_sentence,
            target_triple,
            promise_triple
        )
        # print(prompt)
        response = ollama.chat(
            model = model_name,
            messages = [
                {"role": "user", "content": prompt}
            ]
        )

        # print(response['message']['content'])
        if response['message']['content'].strip().lower() != "no":
            if target_relation not in filtered_rules:
                filtered_rules[target_relation] = []
            filtered_rules[target_relation].append({"relation_premise":primise,"confidence":confidence,"is_reasonable":True})

            logger.info(f"Rule {key} is kept.")

    with open(save_path, 'w') as f:
        json.dump(filtered_rules, f, indent=4)
    logger.info(f"Saved filtered rules to {save_path}.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default='configs/rules_miner/FB15k-237-subset-full-ind.yaml',help="Path to the configuration file")
    args = parser.parse_args()
    logger.info(args.config)
    config = load_config(args.config)
    logger.info(config)
    dataset = config.get('dataset', 'FB15k-237-subset')
    setting = config.get('setting', 'transductive')
    train_size = config.get('train_size', 'full')
    max_path_hops = config.get('max_path_hops', 3)
    model_name = config.get('model', 'llama3.1:70b')

    data_loader = DataLoader(dataset=dataset, setting=setting, train_size=train_size, max_path_hops=max_path_hops)
    data_loader.find_close_paths(save=True)
    # Mine rules
    data_loader.mine_rules(min_support=2, min_confidence=0.1, save=True)
    # Evaluate rules
    evaluate_rule(data_loader,model_name=model_name)

if __name__ == "__main__":
    main()