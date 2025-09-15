from concurrent.futures import ThreadPoolExecutor
import functools
import os
import json
from typing import Literal

from tqdm import tqdm

from run import STRATEGY_MAP, gen_assistant, gen_user, gen_strategy, eval_all, RW_BIAS, print_all
from MCTS import MCTS, Node


tree_root = 'path_to_your_tree_folder'
out_root = 'path_to_your_output_data_folder'

gold_label: Literal['max_Q', 'max_N'] = 'max_Q'
# gold_label: Literal['max_Q', 'max_N'] = 'max_N'

end_min = 0.5
non_perfer_max = 0.5
perfer_N_min = 2

is_extend_non_prefer = False
non_prefer_sim_round = 4
threads = 10

is_cmp = True

key_map = {
        "max_Q": lambda child: child.Q,
        "max_N": lambda child: child.N,
    }

def find_end(node):
    if len(node.children) == 0:
        return None
    for child in sorted(node.children, key=key_map[gold_label], reverse=True):
        if child.end:
            return child
        result = find_end(child)
        if result is not None:
            return result
    return None

def find_all_end(node):
    if len(node.children) == 0:
        return []
    ends = []
    for child in node.children:
        if child.end and key_map[gold_label](child) >= end_min and child.N >= perfer_N_min:
            ends.append(child)
        ends += find_all_end(child)
    return ends

def tag_end_path(end_list):
    in_path = []
    for end in end_list:
        node: Node = end
        while node is not None:
            in_path.append(node)
            node = node.parent
    return in_path


def build_path(mcts: MCTS):
    node = find_all_end(mcts.tree)
    if not node:
        raise ValueError('not end')
    historys = []
    for n in node:
        historys.append(n.build_history()[:-1])
    return historys

def build_compare(mcts: MCTS):
    node_list = find_all_end(mcts.tree)
    in_path_list = tag_end_path(node_list)
    if not node_list:
        raise ValueError('not end')
    historys = []
    for node in node_list:
        history = []
        while node.parent:
            history.insert(0, {"role": "user", "content": node.user})
            supporter_dict = {"role": "assistant", "content": node.assistant, "strategy": node.strategy, "negative": []}
            used_strategy = []
            for child in node.parent.children:
                if child == node or key_map[gold_label](child) > non_perfer_max or not child.is_extend or child in in_path_list:
                    continue
                supporter_dict["negative"].append({"content": child.assistant, "strategy": child.strategy})
                used_strategy.append(child.strategy)
            history.insert(0, supporter_dict)
            node = node.parent
        history.insert(0, {"role": "user", "content": node.user})
        history = history[:-1]
        historys.append(history)
    return historys


def extend_non_prefer(mcts: MCTS, tree_save_path, pic_save_path):
    mcts.gen_user_fn = functools.partial(gen_user, description=mcts.description, scene=mcts.scene)
    mcts.gen_assistant_fn = gen_assistant
    mcts.gen_strategy_fn = gen_strategy
    mcts.eval_all_fn = eval_all
    mcts.rw_bias = RW_BIAS

    node_list = find_all_end(mcts.tree)
    in_path_list = tag_end_path(node_list)
    for node in tqdm(node_list):
        while node.parent:
            extend_flag = True
            for child in node.parent.children:
                if child != node and key_map[gold_label](child) <= non_perfer_max and child.is_extend and child not in in_path_list:
                    extend_flag = False
                    break
            if extend_flag:
                for child in node.parent.children:
                    if child == node or child.is_extend:
                        continue
                    mcts.extend_node(child)
                    if child.end:
                        sim_history = child.build_history()
                        reward = mcts._reward(sim_history, 0)
                    else:
                        sim_history, sim_round, end = mcts._sim(child)
                        reward = mcts._reward(sim_history, sim_round)
                    child.Q = reward
                    mcts.save(tree_save_path)
                    mcts.draw(pic_save_path)
            node = node.parent
    print_all()

if __name__ =='__main__':
    tree_folder = os.path.basename(tree_root)
    out_path = out_root
    os.makedirs(out_path, exist_ok=True)

    if is_cmp and is_extend_non_prefer:
        if threads <= 1:
            for file in os.listdir(tree_root):
                tree_path = os.path.join(tree_root, file)
                mcts = MCTS.load(tree_path)
                extend_non_prefer(mcts, tree_path, os.path.join(out_path, 'pic', os.path.splitext(file)[0]))
        else:
            with ThreadPoolExecutor(max_workers=threads) as executor:
                tree_path_list = [os.path.join(tree_root, file) for file in os.listdir(tree_root)]
                mcts_list = [MCTS.load(path) for path in tree_path_list]
                pic_path_list = [os.path.join(out_path, 'pic', os.path.splitext(file)[0]) for file in os.listdir(tree_root)]
                executor.map(extend_non_prefer, mcts_list, tree_path_list, pic_path_list)
        print_all()

    data_list = []
    for file in os.listdir(tree_root):
        tree_path = os.path.join(tree_root, file)
        mcts = MCTS.load(tree_path)
        if is_cmp:
            datas = build_compare(mcts)
        else:
            datas = build_path(mcts)
        for d in datas:
            data_list.append({
                "description": mcts.description,
                "scene": mcts.scene,
                "iter": mcts.iter,
                "messages": d
            })
    print(len(data_list))

    out_name = f'{tree_folder}_{gold_label}{"_cmp" if is_cmp else ""}{"_extend" if is_extend_non_prefer and is_cmp else ""}'
    with open(os.path.join(out_path, f'{out_name}.json'), 'w', encoding='utf-8') as f:
        json.dump(data_list, f, ensure_ascii=False, indent=2)


