import os
import json
import math
import time
from typing import Callable, Literal, Union
import pickle
import functools
from copy import deepcopy

from tqdm import trange
from graphviz import Digraph
import numpy as np

# FORCE_EXTEND = True
FORCE_EXTEND = False

SHORT_MAP = {
    'Emotional Validation': 'Emotional Valid',
    'Affirmation': 'Affirmation',
    'Collaborative Planning': 'C Planning',
    'Empathetic Statements': 'Empa Statements',
    'Avoid Judgment and Criticism': 'Criticism',
    'Provide Different Perspectives': 'D Perspectives',
    'Reframe Negative Thoughts': 'Reframe NT',
    'Share Information': 'S Information',
    'start': 'start',
}



class Node:
    def __init__(self, parent: "Node", strategy, strategy_score: float, c) -> None:
        self.parent = parent
        self.children: list["Node"] = []
        self.strategy = strategy
        self.strategy_score = strategy_score
        self.end = False
        self.c = c
        
        self.N = 0
        self.Q = 0
        self.PUCB = 0
        self.is_extend = False
        self.bad_list = []

    def extend(self, assistant, user, end):
        self.is_extend = True
        self.assistant = assistant
        self.user = user
        self.end = end

    @property
    def id(self) -> str:
        return str(id(self))
    
    @property
    def label(self) -> str:
        if not self.is_extend:
            return f'PUCB:{self.PUCB:.4f}\nscore:{self.strategy_score:.4f}\n{SHORT_MAP[self.strategy]}\nQ:{self.Q:.4f}\nN:{self.N}'
        return f'PUCB:{self.PUCB:.4f}\nscore:{self.strategy_score:.4f}\n{SHORT_MAP[self.strategy]}\nQ:{self.Q:.4f}\nN:{self.N}'

    def _update_PUCB(self):
        self.PUCB = self.Q + self.c * self.strategy_score * math.sqrt(self.parent.N) / (self.N + 1)
        return self

    def backward(self, reward):
        self.Q = (self.Q * self.N + reward) / (self.N + 1)
        self.N += 1
        for child in self.children:
            child._update_PUCB()

        if self.parent is None:
            return
        self.parent.backward(reward)

    def build_history(self):
        if self.parent is None:
            return [
                {"role": "supporter", "content": self.assistant, "strategy": self.strategy},
                {"role": "user", "content": self.user},
            ]
        return self.parent.build_history() + [
            {"role": "supporter", "content": self.assistant, "strategy": self.strategy},
            {"role": "user", "content": self.user},
        ]
    
    def count_end(self):
        if self.end:
            return 1
        num = 0
        for c in self.children:
            num += c.count_end()
        return num



class MCTS:
    def __init__(self, init_assistant='', c=10, sim_max_round: Union[int, Literal['end']]='end', gen_strategy_fn=None, gen_assistant_fn=None, gen_user_fn=None, eval_all_fn=None, rw_bias=0) -> None:
        self.init_assistant = init_assistant
        self.c = c
        self.sim_max_round = sim_max_round
        self.gen_strategy_fn: Callable[[list], tuple[list[dict], bool]] = gen_strategy_fn
        self.gen_assistant_fn: Callable[[list, str, str], tuple[str, float]] = gen_assistant_fn
        self.raw_gen_user_fn: Callable[[list, str, str, bool], tuple[str, bool]] = gen_user_fn
        self.eval_all_fn: Callable[[list], float] = eval_all_fn
        self.rw_bias = rw_bias

    def _reward(self, sim_history, sim_round):
        r = self.eval_all_fn(sim_history, sim_round)
        return r + self.rw_bias
    
    def extend_node(self, node: Node):
        history = node.parent.build_history()
        assistant = self.gen_assistant_fn(history, node.strategy)
        user, end = self.gen_user_fn(history + [{"role": "supporter", "content": assistant, "strategy": node.strategy}])
        node.extend(assistant=assistant, user=user, end=end)

    def select(self):
        node = self.tree
        while not len(node.children) == 0:
            node = max(node.children, key=lambda child: child.PUCB)
        if not node.is_extend:
            self.extend_node(node)
        return node
    
    def expand(self, node: Node):
        history = node.build_history()
        score_list = self.gen_strategy_fn(history)
        for score in score_list:
            if FORCE_EXTEND:
                assistant, res_score, bad_list = self.gen_assistant_fn(history, score['strategy'])
                user, end = self.gen_user_fn(history + [{"role": "supporter", "content": assistant, "strategy": score['strategy']}])
                end = end
                child = Node(parent=node, strategy=score['strategy'], strategy_score=score['score'], c=self.c)
                child.extend(assistant=assistant, user=user, end=end)
            else:
                child = Node(parent=node, strategy=score['strategy'], strategy_score=score['score'], c=self.c)
            node.children.append(child)
        self.draw()

    def _sim(self, node: Node):
        history = node.build_history()

        for i in range(1000 if self.sim_max_round == 'end' else self.sim_max_round):
            score_list = self.gen_strategy_fn(history)
            score = max(score_list, key=lambda x: x['score'])

            assistant = self.gen_assistant_fn(history, score['strategy'])
            history.append({"role": "supporter", "content": assistant, "strategy": score['strategy']})
            user, end = self.gen_user_fn(history)
            history.append({"role": "user", "content": user})
            if end:
                break
        return history, i + 1, end
    
    def simulate_and_backpropagate(self, node: Node):
        # print('\nsim!')
        child = max(node.children, key=lambda child: child.strategy_score)
        if not child.is_extend:
            self.extend_node(child)
        if child.end:
            sim_history = child.build_history()
            child.backward(self._reward(sim_history, 0))
            return
        sim_history, sim_round, end = self._sim(child)
        child.backward(self._reward(sim_history, sim_round))

    def update(self):
        selected = self.select()
        if not selected.end:
            self.expand(selected)
            self.simulate_and_backpropagate(selected)
            self.draw()
        else:
            sim_history = selected.build_history()
            selected.backward(self._reward(sim_history, 0))
            self.draw()

    def run(self, description, scene, min_iter, min_end, max_iter, max_end, tmp_path, json_data=None):
        self.min_iter = min_iter
        self.min_end = min_end
        self.max_iter = max_iter
        self.max_end = max_end
        self.gen_user_fn: Callable[[list,bool], tuple[str, bool]] = functools.partial(self.raw_gen_user_fn, description=description, scene=scene)
        self.tmp_path = tmp_path
    
        tree_tmp_path = f'{tmp_path}.pkl'
        if os.path.exists(tree_tmp_path):
            self.load_tmp(tree_tmp_path)
        elif json_data:
            self.description = json_data['description']
            self.scene = json_data['scene']
            self.iter = 0
            self.build_from_json(json_data)
            self.draw()
        else:
            self.description = description
            self.scene = scene
            self.iter = 0
            user, end = self.gen_user_fn([{"role": "supporter", "content": self.init_assistant}])
            self.tree = Node(parent=None, strategy="start", strategy_score=1, c=self.c)
            self.tree.extend(assistant=self.init_assistant, user=user, end=end)

        start_i = self.iter
        # print(start_i)
        for i in trange(start_i, max_iter, initial=start_i, total=max_iter):
            end_num = self.tree.count_end()
            sel_end = self.select().end
            if self.iter >= min_iter:
                print('min iter reached, ', f'{self.iter=}, {min_iter=}')
            if end_num >= min_end:
                print('min end reached, ', f'{end_num=}, {min_end=}')
            if not sel_end and self.iter >= min_iter and end_num >= min_end:
                print('min end and min iter reached, ', f'{self.iter=}, {end_num=}')
                return
            if not sel_end and end_num >= max_end:
                print('max end reached, ', f'{self.iter=}, {end_num=}')
                return
            self.update()
            print(f'\n{"-"*40}\niter {i}: {time.asctime()}\n{"-"*40}\n')
            self.iter = i + 1
            self.save(tree_tmp_path)
        print('max iter reached')


    def draw(self, save_path=None, format='png'):
        if save_path == None:
            save_path = self.tmp_path
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        color_map = {
            "Emotional Validation": "#A1A9D0",
            "Affirmation": "#F0988C",
            "Collaborative Planning": "#B883D4",
            "Empathetic Statements": "#9E9E9E",
            "Avoid Judgment and Criticism": "#CFEAF1",
            "Provide Different Perspectives": "#C4A5DE",
            "Reframe Negative Thoughts": "#F6CAE5",
            "Share Information": "#96CCCB",
            "start": "#FFFFFF"
        }
        def add_nodes_edges(graph: Digraph, node: Node):
            for child in node.children:
                dot.node(child.id, label=child.label, style='filled', fillcolor=color_map[child.strategy], shape='box' if child.end else ('ellipse' if child.is_extend else 'octagon'))
                graph.edge(node.id, child.id)
                add_nodes_edges(graph, child)
        dot = Digraph(comment='Tree Visualization')
        dot.node(self.tree.id, label=self.tree.label, style='filled', fillcolor=color_map[self.tree.strategy])
        add_nodes_edges(dot, self.tree)
        dot.render(save_path, format=format, view=False)

    def save(self, path):
        save_dict = {"c": self.c, "sim_max_round": self.sim_max_round, "description": self.description, "scene": self.scene, "iter": self.iter, "tree": self.tree}

        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(save_dict, f)

    def load(path):
        with open(path, 'rb') as f:
            save_dict = pickle.load(f)
            mcts = MCTS(None, c=save_dict['c'], sim_max_round=save_dict['sim_max_round'])
            mcts.tree = save_dict['tree']
            mcts.description = save_dict['description']
            mcts.scene = save_dict['scene']
            mcts.iter = save_dict['iter']
            return mcts
        
    def load_tmp(self, path):
        with open(path, 'rb') as f:
            save_dict = pickle.load(f)
        self.tree = save_dict['tree']
        self.description = save_dict['description']
        self.scene = save_dict['scene']
        self.iter = save_dict['iter']
        
    def get_best_json(self):
        # best Q
        node = self.tree
        while not len(node.children) == 0:
            node = max(node.children, key=lambda child: child.Q)
        return node.build_history()
    
    def build_from_json(self, json_data):
        if json_data['messages'][0]['role'] == 'user':
            self.tree = Node(parent=None, strategy="start", strategy_score=1, c=self.c)
            self.tree.extend(assistant=self.init_assistant, user=json_data['messages'][0]['content'], end=False)
            self.tree.Q = 4 + self.rw_bias
            self.tree.N = 100
            json_data['messages'].pop(0)
        else:
            self.tree = Node(parent=None, strategy=json_data['messages'][0]['strategy'], strategy_score=1, c=self.c)
            self.tree.extend(assistant=json_data['messages'][0]['content'], user=json_data['messages'][1]['content'], end=False)
            self.tree.Q = 4 + self.rw_bias
            self.tree.N = 100
            json_data['messages'].pop(0)
            json_data['messages'].pop(0)
        node = self.tree
        for i in range(0, len(json_data['messages']), 2):
            scores = self.gen_strategy_fn(json_data['messages'][:i])
            if all(s['strategy'] != json_data['messages'][i]['strategy'] for s in scores):
                for s in scores:
                    s['score'] *= 0.9
                scores.append({'strategy': json_data['messages'][i]['strategy'], 'score': 0.1})
            for s in scores:
                if s['strategy'] == json_data['messages'][i]['strategy']:
                    new_node = Node(parent=node, strategy=s['strategy'], strategy_score=s['score'], c=self.c)
                    if len(json_data['messages']) > i+1:
                        new_node.extend(assistant=json_data['messages'][i]['content'], user=json_data['messages'][i+1]['content'], end=False)
                    else:
                        new_node.extend(assistant=json_data['messages'][i]['content'], user=None, end=False)
                    new_node.Q = 4 + self.rw_bias
                    new_node.N = 100
                    node.children.append(new_node._update_PUCB())
                else:
                    node.children.append(Node(parent=node, strategy=s['strategy'], strategy_score=s['score'], c=self.c)._update_PUCB())
            node = new_node
        new_node.end = True

            
