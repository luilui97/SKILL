import os
import json
import random
import copy
import numpy as np
from collections import defaultdict, deque, Counter
from sentence_transformers import SentenceTransformer
from prompt_templates import BASE_REASON_PROMPT, RULE_PATH_PROMPT, RULE_NE_PATH_PROMPT, PATH_PROMPT
from tqdm import tqdm
from utils import setup_logger

logger = setup_logger()

class DataLoader:
    def __init__(self, dataset="FB15k-237-subset", setting="inductive", train_size="full", max_path_hops=3):
        self.dataset = dataset
        self.dataset_name = dataset.split("-")[0]
        self.dataset_path = f"datasets/{dataset}" + ("-inductive" if setting=="inductive" else "")
        self.train_size = train_size
        self.setting = setting
        
        self.test_batch_size = 50                                    # 测试集中每50个sample为一个batch，并计算MRR和Hits@1
        self.max_path_hops = max_path_hops
        self.max_reason_paths = 6
        self.max_reason_rules = 6
        
        self.entity2text = self._load_text_file("entity2text.txt")
        self.relation2text = self._load_text_file("relation2text.txt")
        if "FB15k-237" in self.dataset:
            self.entity2description = self._load_text_file("entity2textlong.txt") # FB15k-237的实体的详细描述
        else:
            self.entity2description = copy.deepcopy(self.entity2text)  # 其他数据集的实体描述和实体名称相同
        
        self.train_set = self._load_triples(f"train_{self.train_size}.txt")
        self.path_set = self._load_triples("inductive_graph.txt") if setting=="inductive" else self.train_set
        self.valid_set = self._load_triples(f"valid.txt")
        self.test_set_head = self._load_triples(f"ranking_head.txt")
        self.test_set_tail = self._load_triples(f"ranking_tail.txt")
        self.test_set = self.test_set_head + self.test_set_tail
        
        self.relation2headtail_dict = self._load_relation2headtail_dict(self.path_set)
        self.entity2relationtail_dict = self._load_entity2relationtail_dict(self.path_set)
        # self.entity2relationtail_dict, self.entity2relationtail_dict_forward = self._load_entity2relationtail_dict(self.path_set)
        self.relation_degree_dict = self._load_relation_degree_dict(self.path_set)
        self.close_path_dict = defaultdict(list)

        self.rule_counter = Counter()
        self.path_counter = Counter()
        self.raw_rules = {}

        self.embedding_model = SentenceTransformer(
            model_name_or_path='modelscope/BAAI/bge-small-en-v1.5',
            device="cuda"
        )

    def _load_text_file(self, filename):
        filepath = f"{self.dataset_path}/{filename}"
        with open(filepath, "r", encoding="utf-8") as file:
            return dict(line.strip().split('\t', 1) for line in file if line.strip())

    def _load_triples(self, filename):
        filepath = f"{self.dataset_path}/{filename}"
        with open(filepath, "r", encoding="utf-8") as file:
            return [line.strip().split('\t') for line in file if line.strip()]

    def _load_relation2headtail_dict(self, triple_set):
        relation2headtail_dict = defaultdict(list)
        for head, relation, tail in triple_set:
            relation2headtail_dict[relation].append([head, tail])
        return relation2headtail_dict
    
    def _load_entity2relationtail_dict(self, triple_set):
        entity2relationtail_dict = defaultdict(list)
        for head, relation, tail in triple_set:
            entity2relationtail_dict[head].append((relation, tail, 1))
            entity2relationtail_dict[tail].append((relation, head, -1))
        return entity2relationtail_dict

    def _load_relation_degree_dict(self, triple_set):
        relation_degree_dict = defaultdict(int)
        for _, relation, _ in triple_set:
            relation_degree_dict[relation] += 1
        return relation_degree_dict


    # 输入head entity和tail entity，使用bfs遍历搜索所有close_paths
    def bfs_paths(self, start, goal):
        queue = deque([(start, [], 0, set([start]))])
        paths = []
        while queue:
            current, path, hops, visited = queue.popleft()
            if hops < self.max_path_hops:
                for relation, neighbor, direction in self.entity2relationtail_dict[current]:
                    if direction == 1:
                        new_path = path + [(current, relation, neighbor)]
                    else:
                        new_path = path + [(neighbor, relation, current)]
                    if neighbor == goal:
                        paths.append(new_path)
                    elif neighbor not in visited:
                        queue.append((neighbor, new_path, hops + 1, visited | set([neighbor])))
        return paths
    
    def dfs_paths(self, start, goal):
        stack = [(start, [], 0, set([start]))]
        paths = []
        while stack:
            current, path, hops, visited = stack.pop()
            if hops < self.max_path_hops:
                for relation, neighbor, direction in self.entity2relationtail_dict[current]:
                    if direction == 1:
                        new_path = path + [(current, relation, neighbor)]
                    else:
                        new_path = path + [(neighbor, relation, current)]
                    if neighbor == goal:
                        paths.append(new_path)
                    elif neighbor not in visited:
                        stack.append((neighbor, new_path, hops + 1, visited | set([neighbor])))
        return paths

    def find_close_paths(self, save=True):
        """
        Find close paths in the dataset using BFS.
        
        Returns:
            dict: A dictionary with pairs of entities as keys and their close paths as values.
        """
        save_dir = f"{self.dataset_path}/paths_{self.max_path_hops}_hop"
        
        if self.setting == "inductive":
            close_path_file = f"{save_dir}/close_path.json"
        else:
            close_path_file = f"{save_dir}/close_path_train_size_{self.train_size}.json"
        if os.path.exists(close_path_file):
            with open(close_path_file, "r", encoding="utf-8") as f:
                self.close_path_dict = json.load(f)
            return

        if self.setting == "transductive":
            
            for triple in tqdm(self.train_set, desc=f"Finding close paths in {self.dataset} - {self.setting} - {self.train_size}"):
                head, relation, tail = triple
                paths = list(self.bfs_paths(head, tail))
                for path_pair in paths:
                    if (head, relation, tail) not in path_pair:
                        path_pair.insert(0, (head, relation, tail))
                self.close_path_dict[f"{relation}"].append(paths)
        
        elif self.setting == "inductive":
            for triple in tqdm(self.path_set, desc=f"Finding close paths in {self.dataset} - {self.setting} - {self.train_size}"):
                head, relation, tail = triple
                paths = list(self.bfs_paths(head, tail))
                for path_pair in paths:
                    if (head, relation, tail) not in path_pair:
                        path_pair.insert(0, (head, relation, tail))
                self.close_path_dict[f"{relation}"].append(paths)
        
        logger.info(f"Total close paths found: {len(self.close_path_dict)}")
        if save:
            os.makedirs(save_dir, exist_ok=True)
            with open(close_path_file, "w", encoding="utf-8") as f:
                json.dump(self.close_path_dict, f, ensure_ascii=False, indent=4)



    def path_walker_dfs(self, head, current_path, current_entity, max_hops, visted_entities):
        if len(current_path) > 0:
            path_key = tuple(current_path)
            self.path_counter[path_key] += 1
        
        if len(current_path) == max_hops:
            return
        
        for relation, tail, _ in self.entity2relationtail_dict.get(current_entity, []):
            if tail not in visted_entities:
                self.path_walker_dfs(head, current_path+[relation], tail, max_hops, visted_entities | set([tail]))


    def calculate_paths_count(self):
        """
        Calculate the support of different paths in the dataset.
        
        Returns:
            dict: A dictionary with path keys as keys and their support counts as values.
        """
        for entity in tqdm(self.entity2relationtail_dict, desc=f"Calculating paths support in {self.dataset} - {self.setting} - {self.train_size}"):
            for r1, tail, _ in self.entity2relationtail_dict[entity]:
                self.path_walker_dfs(entity, [r1], tail, self.max_path_hops, set([entity, tail]))
        
    
    def mine_rules(self, min_support=2, min_confidence=0.1, save=True):
        """
        Mine rules from the dataset using DFS.
        
        Args:
            min_support (int): Minimum support for a rule to be considered.
            min_confidence (float): Minimum confidence for a rule to be considered.
            max_hops (int): Maximum hops for the DFS search.
        
        Returns:
            list: A sorted list of rules with their support and confidence.
        """
        save_dir = f"{self.dataset_path}/rules_{self.max_path_hops}_hop"
        
        if self.setting == "inductive":
            rule_file = f"{save_dir}/rules.json"
        else:
            rule_file = f"{save_dir}/rules_train_size_{self.train_size}.json"
        if os.path.exists(rule_file):
            with open(rule_file, "r", encoding="utf-8") as f:
                self.raw_rules = json.load(f)
            return

        rule_instance = {}

        for relation, closed_paths in tqdm(self.close_path_dict.items(), desc=f"Mining rules in {self.dataset} - {self.setting} - {self.train_size}"):
            for paths in closed_paths:
                for path in paths:
                    if len(path) > 1:
                        path_key = tuple([triple[1] for triple in path[1:]]) 
                        self.rule_counter[(path_key, relation)] += 1
                        if ' - '.join(path_key) + f' -> {relation}' not in rule_instance:
                            rule_instance[' - '.join(path_key) + f' -> {relation}'] = []
                        rule_instance[' - '.join(path_key) + f' -> {relation}'].append(path) 


        self.calculate_paths_count()               
        
        candidate_rules = []
        logger.info(f"Total paths found: {len(self.path_counter)}")
        logger.info(f"Total rules found: {len(self.rule_counter)}")
        for (path_key, relation), support in self.rule_counter.items():
            total = self.path_counter[path_key]
            confidence = support / total
            if support >= min_support:
                candidate_rules.append((path_key, relation, support, round(confidence, 3)))
        candidate_rules = sorted(candidate_rules, key=lambda x: (-x[2], -x[3]))

        for path_key, relation, support, confidence in candidate_rules:
            key = ' - '.join(path_key) + f' -> {relation}'
            if key not in self.raw_rules:  
                self.raw_rules[key] = {}
            self.raw_rules[key]['relation'] = relation 
            self.raw_rules[key]['relation_premise'] = path_key
            self.raw_rules[key]['support'] = support
            self.raw_rules[key]['confidence'] = confidence
            self.raw_rules[key]['path_instances'] = rule_instance.get(key, []) 

        
        if save:
            os.makedirs(save_dir, exist_ok=True)
            with open(rule_file, "w", encoding="utf-8") as f:
                json.dump(self.raw_rules, f, ensure_ascii=False, indent=4)

    
    def close_path_finder(self, triple):
        head, relation, tail = triple
        head_tail = f"{head}-{tail}"
        close_paths = self.close_path_dict[head_tail]

        if close_paths:
            path_degrees = []
            for path in close_paths:
                degree_sum = sum(self.relation_degree_dict[rel] for _, rel, _ in path)
                path_degrees.append((degree_sum, path))
            path_degrees.sort(key=lambda x: x[0])
            
            top_paths = [path for _, path in path_degrees[:self.max_reason_paths]]
            top_paths.reverse()
            return top_paths

        return []
    
    
    def rule_path_finder(self, triple, rules):
        """
        Find rules corresponding to the given triple.
        """
        head, relation, tail = triple
        close_paths = list(self.bfs_paths(head, tail))
        if close_paths:
            path_degrees = []
            for path in close_paths:
                degree_sum = sum(self.relation_degree_dict[rel] for _, rel, _ in path)
                path_degrees.append((degree_sum, path))
            path_degrees.sort(key=lambda x: x[0])
            
            top_paths = [path for _, path in path_degrees[:self.max_reason_paths]]
            top_paths.reverse()

            rule_candidates = []
            rules_for_triple = rules.get(relation, [])
            if not rules_for_triple:
                return top_paths,[]
            for path in close_paths:
                promises = [rel for _, rel, _ in path]
                for rule in rules_for_triple:
                    satisfied_promised = set(rule["relation_premise"]).intersection(set(promises))
                    if len(satisfied_promised) >0:
                        if rule not in rule_candidates:
                            rule['satisfied_ratio'] = len(satisfied_promised) / len(rule["relation_premise"])
                            rule_candidates.append(rule)
            rule_candidates.sort(key=lambda x: x.get("satisfied_ratio", 0.0)*x.get("confidence", 0.0))
            top_rule_candidates = rule_candidates[:self.max_reason_rules]
            return top_paths, top_rule_candidates
            
        return [],[]
    
    
    def rule_path_sentence(self, rule_paths, relation):
        """
        Convert rule paths to sentence format.
        """
        if not rule_paths:
            return ""
        
        sentences = []
        count = 1
        for rule in rule_paths:
            promises = rule["relation_premise"]
            if self.dataset == "FB15k-237-subset":
                premise_sentences = [
                    f"is the {premise.split('/')[-1]} of {premise.split('/')[2]}"
                    for premise in promises
                ]
                relation_sentence = f"is the {relation.split('/')[-1]} of {relation.split('/')[2]}"
                rule_sentence = " | ".join(premise_sentences) + f" -> {relation_sentence}"
            elif self.dataset == "WN18RR-subset":
                rule_sentence = " | ".join([self.relation2text[premise] for premise in promises])+ f" -> {self.relation2text[relation]}"
            elif self.dataset == "NELL-995-subset":
                rule_sentence = " | ".join([self.relation2text[premise] for premise in promises])+ f" -> {self.relation2text[relation]}"
            sentence = f"Rule {count}:\n {rule_sentence}\n"
            sentences.append(sentence)
            count += 1
        
        return "\n".join(sentences)
    
    def closed_path_sentence(self, close_paths):
        """
        Convert close paths to sentence format.
        """
        if not close_paths:
            return ""
        
        sentences = []
        count = 1
        for single_path in close_paths:
            path_sentence = " -> ".join(self.triple_to_sentence(triple) for triple in single_path)
            sentence = f"Path {count}:\n {path_sentence}"
            sentences.append(sentence)
            count += 1
        
        return "\n".join(sentences)

    def linearize_triple(self, triple):
        return f"({self.entity2text[triple[0]]}, {self.relation2text[triple[1]]}, {self.entity2text[triple[2]]})"
    
    def triple_to_sentence(self, triple):
        head, relation, tail = triple
        if self.dataset == "FB15k-237-subset":
            head_property = relation.split('/')[2]
            tail_property = relation.split('/')[-1]
            return f"('{self.entity2text[tail]}' is the {tail_property} of {head_property} '{self.entity2text[head]}')"
        elif self.dataset == "WN18RR-subset":
            return f"('{self.entity2text[head]}', {self.relation2text[relation]}, '{self.entity2text[tail]}')"
        elif self.dataset == "NELL-995-subset":
            return f"('{self.entity2text[head]}', {self.relation2text[relation]}, '{self.entity2text[tail]}')"
    
    
    def build_vanilla_prompt(self, triple):
        return BASE_REASON_PROMPT.format(test_triple=self.triple_to_sentence(triple))
    
    def build_rule_path_prompt(self, triple, rules):
        """
        Build a prompt for the rule path reasoning.
        """
        head, relation, tail = triple

        close_paths, rule_candidates = self.rule_path_finder(triple, rules)
        rule_paths_sentence = self.rule_path_sentence(rule_candidates, relation)
        closed_paths_sentence = self.closed_path_sentence(close_paths)
        head_entity_text = self.entity2text.get(head,"")
        tail_entity_text = self.entity2text.get(tail,"")

        return RULE_PATH_PROMPT.format(
            head_entity=head_entity_text,
            tail_entity=tail_entity_text,
            reasoning_rules=rule_paths_sentence,
            reasoning_paths=closed_paths_sentence,
            test_triple=self.triple_to_sentence(triple)
        )
    
    def build_path_prompt(self, triple, rules):
        """
        Build a prompt for the rule path reasoning.
        """
        head, relation, tail = triple

        close_paths, rule_candidates = self.rule_path_finder(triple, rules)
        # rule_paths_sentence = self.rule_path_sentence(rule_candidates, relation)
        closed_paths_sentence = self.closed_path_sentence(close_paths)
        head_entity_text = self.entity2text.get(head,"")
        tail_entity_text = self.entity2text.get(tail,"")

        return PATH_PROMPT.format(
            head_entity=head_entity_text,
            tail_entity=tail_entity_text,
            reasoning_paths=closed_paths_sentence,
            test_triple=self.triple_to_sentence(triple)
        )
       
    def build_rule_ne_path_prompt(self, triple, rules):
        """
        Build a prompt for the rule path reasoning with neighbor triples.
        """
        head, relation, tail = triple
        close_paths, rule_candidates = self.rule_path_finder(triple, rules)
        rule_paths_sentence = self.rule_path_sentence(rule_candidates, relation)
        closed_paths_sentence = self.closed_path_sentence(close_paths)
        head_entity_text = self.entity2text.get(head,"")
        tail_entity_text = self.entity2text.get(tail,"")
        neighbor_triples = self.neighbor_triple_finder_rule(triple,rules)

        return RULE_NE_PATH_PROMPT.format(
            head_entity=head_entity_text,
            tail_entity=tail_entity_text,
            neighbor_triples='\n'.join(neighbor_triples),
            reasoning_rules=rule_paths_sentence,
            reasoning_paths=closed_paths_sentence,
            test_triple=self.triple_to_sentence(triple)
        )

    def get_test_batches(self):
        return [self.test_set[i:i + self.test_batch_size] for i in range(0, len(self.test_set), self.test_batch_size)]
    

    # path_reasoning里面的neighbor triple，尽可能找到与当前triple相关的neighbor triple
    def neighbor_triple_finder(self, triple):
        head, relation, tail = triple
        head_triples = self.entity2relationtail_dict[head]
        tail_triples = self.entity2relationtail_dict[tail]



        triple_sentence = self.triple_to_sentence(triple)
        head_sentences = [self.triple_to_sentence((head, rel, t)) if direction == 1 else self.triple_to_sentence((t, rel, head))
                          for rel, t, direction in head_triples]
        tail_sentences = [self.triple_to_sentence((tail, rel, h)) if direction == 1 else self.triple_to_sentence((h, rel, tail))
                          for rel, h, direction in tail_triples]
        
        all_head_sentences = [triple_sentence] + head_sentences
        all_tail_sentences = [triple_sentence] + tail_sentences
        
        each_count = self.max_reason_paths // 2
        
        top_head_sentences = head_sentences
        top_tail_sentences = tail_sentences

        if len(head_sentences) > each_count:
            head_embeddings = self.embedding_model.encode(all_head_sentences, normalize_embeddings=True)
            head_similarity = head_embeddings[0] @ head_embeddings[1:].T
            top_head_indices = np.argsort(-head_similarity)[:each_count]
            top_head_sentences = [head_sentences[i] for i in top_head_indices]

        if len(tail_sentences) > each_count:
            tail_embeddings = self.embedding_model.encode(all_tail_sentences, normalize_embeddings=True)
            tail_similarity = tail_embeddings[0] @ tail_embeddings[1:].T
            top_tail_indices = np.argsort(-tail_similarity)[:each_count]
            top_tail_sentences = [tail_sentences[i] for i in top_tail_indices]
        
        return top_head_sentences + top_tail_sentences
    
    # rule guided neighbor triple finder, 会根据规则筛选出与triple相关的neighbor triple
    def neighbor_triple_finder_rule(self, triple, rules):
        head, relation, tail = triple
        head_triples = self.entity2relationtail_dict[head]
        tail_triples = self.entity2relationtail_dict[tail]
        

        rules_for_triple = rules.get(relation, [])
        # Flatten all relation premises from rules_for_triple into a set
        all_rule_head = set(rel for rule in rules_for_triple for rel in rule["relation_premise"])
        if rules_for_triple:
            new_head_triples = []
            new_tail_triples = []
            for rel, t, direction in head_triples:
                if rel in all_rule_head:
                    new_head_triples.append((rel, t, direction))
            for rel, h, direction in tail_triples:
                if rel in all_rule_head:
                    new_tail_triples.append((rel, h, direction))
            if new_head_triples:
                head_triples = new_head_triples
            if new_tail_triples:
                tail_triples = new_tail_triples

        triple_sentence = self.triple_to_sentence(triple)
        head_sentences = [self.triple_to_sentence((head, rel, t)) if direction == 1 else self.triple_to_sentence((t, rel, head))
                          for rel, t, direction in head_triples]
        tail_sentences = [self.triple_to_sentence((tail, rel, h)) if direction == 1 else self.triple_to_sentence((h, rel, tail))
                          for rel, h, direction in tail_triples]
        
        all_head_sentences = [triple_sentence] + head_sentences
        all_tail_sentences = [triple_sentence] + tail_sentences
        
        each_count = self.max_reason_paths // 2
        
        top_head_sentences = head_sentences
        top_tail_sentences = tail_sentences

        if len(head_sentences) > each_count:
            head_embeddings = self.embedding_model.encode(all_head_sentences, normalize_embeddings=True)
            head_similarity = head_embeddings[0] @ head_embeddings[1:].T
            top_head_indices = np.argsort(-head_similarity)[:each_count]
            top_head_sentences = [head_sentences[i] for i in top_head_indices]

        if len(tail_sentences) > each_count:
            tail_embeddings = self.embedding_model.encode(all_tail_sentences, normalize_embeddings=True)
            tail_similarity = tail_embeddings[0] @ tail_embeddings[1:].T
            top_tail_indices = np.argsort(-tail_similarity)[:each_count]
            top_tail_sentences = [tail_sentences[i] for i in top_tail_indices]
        
        return top_head_sentences + top_tail_sentences
    
    # 负采样：对于正样本triple，分别破坏head, relaton和tail，并为它们随机采样。
    def neg_sampling(self, pos_triple, count):
        head, relation, tail = pos_triple
        
        entities = set()
        for triple in self.path_set:
            entities.add(triple[0])
            entities.add(triple[2])
        
        candidate_entities = entities - {head, tail}
        seen_triples = {tuple(triple) for triple in self.path_set}
        negative_samples = []
        
        # 破坏head
        for _ in range(count):
            while True:
                new_head = random.choice(list(candidate_entities))
                if (new_head, relation, tail) not in seen_triples:
                    seen_triples.add((new_head, relation, tail))
                    negative_samples.append((new_head, relation, tail))
                    break
        
        # 破坏tail
        for _ in range(count):
            while True:
                new_tail = random.choice(list(candidate_entities))
                if (head, relation, new_tail) not in seen_triples:
                    seen_triples.add((head, relation, new_tail))
                    negative_samples.append((head, relation, new_tail))
                    break
        
        return negative_samples

if __name__ == "__main__":
    dataset = "FB15k-237-subset"
    setting = "transductive"
    train_size = "full"
    
    data_loader = DataLoader(dataset=dataset, setting=setting, train_size=train_size)
    data_loader.find_close_paths(save=True)
    # Mine rules
    data_loader.mine_rules(min_support=2, min_confidence=0.1, save=True)