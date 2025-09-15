import re
import os
import json
import math
import random
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from tqdm import tqdm

from util import retry, history_to_str, generate
from MCTS import MCTS


CHARA_DETAIL = '''
You're engaged in a face-to-face conversation, with each of your sentences being fairly short and informal. Most of the time you speak less than 25 words at a time.
'''

STRATEGY_MAP = {
    'Emotional Validation': 'Acknowledge and validate the User’s emotions without judgment.',

    'Affirmation': 'Provide positive reinforcement and encouragement to uplift the User’s spirits.',

    'Collaborative Planning': 'Work together with the User to develop an action plan.',

    'Empathetic Statements': 'Express understanding and empathy towards the User’s experiences.',
    'Avoid Judgment and Criticism': 'It’s important to create a non-judgmental and safe space for the User to express their emotions without fear of criticism. Refrain from passing judgment or being overly critical of their experiences or choices.',
    'Provide Different Perspectives': 'Offer alternative ways of looking at the situation to help the User gain new insights.',
    'Reframe Negative Thoughts': 'Help the User reframe negative thoughts into more positive or realistic ones.',
    'Share Information': 'Provide educational or factual information about emotions, coping mechanisms, or self-care practices.',
}

STRATEGY_KEY_MAP = {k.lower(): k for k in STRATEGY_MAP}

USER_SYS_PROMPT = '''
You are someone who has been feeling down due to recent circumstances. You can add details about yourself in the conversation.

You are currently in a bad emotion due to the following situation:
{situation}

You have sought out a supporter to talk to, in the hopes of alleviating your feelings of {emotion}.

You're engaged in a face-to-face conversation, with each of your sentences being fairly short and informal.Please try to simulate real human-spoken conversation as much as possible, don't act like a machine, don't mention that you are an AI assistant. Don't always repeat or say 'thanks'.
In the beginning, you shouldn't reveal all the information at once and follow up with further clarification only when asked. You speak less than 20 words at a time.

You will get the conversation between you and the supporter.
'''

USER_QUERY = '''
{chat_history}

Output your next sentence directly, if you feel the conversation should end and you won't continue, output a special word </end/>. Do not add other extraneous prefixes and control characters.
'''

SUPPORTER_SYS_PROMPT = '''
You are a emotion supporter, providing emotional support to seekers. You are now going to help alleviate a user's emotions. Please try to simulate real human-spoken conversation as much as possible, don't act like a machine, don't mention that you are an AI assistant.

## character setting

{character}

You will get the conversation between you and the user, and your next strategy. Output your next sentence directly, do not add other extraneous prefixes and control characters.
'''

SUPPORTER_QUERY = '''
{chat_history}

You will use the strategy of "{strategy}" to respond to the user, below are the detail of the strategy:
{strategy_detail}

Here is your output:
'''

STRATEGY_SYS_PROMPT = '''
You will play the role of a psychologist who is well versed in emotional support. You are proficient in various support techniques and strategies and understand how to apply them.
'''

STRATEGY_QUERY = '''
A supporter is working to alleviate a user's negative emotions. Below is the conversation between supporter and the visitor:
{chat_history}

The supporter will use one of the following strategies in response. Here are the available strategies along with their detailed descriptions:
{stratrgy_str}

Please rate the feasibility of all the available strategies on a scale from 0 to 10, where a higher number indicates better feasibility.

IMPORTANT: Don't use the same strategy consecutively. Don't always use Collaborative Planning.

Output the scores for avaliable strategies in a list with JSON-like format: `{{strategy: score}}`, and use backticks to enclose the JSON structure.
Think before output your answer, don't include any strategy not in available strategies. The strategy should be completely consistent with the above.

Here is your reason and output:
'''


SUPPORTER_START = "Hello, I'm your supporter. Welcome. How can I support you today?"


EVAL_ROUND_PROMPT = '''You will play the role of a psychologist who is well versed in emotional support. There will be a dialog between the help seeker (i.e., the person seeking support) and the supporter (i.e., the person providing support). The dialog will consist of the help-seeker asking the supporter for help with some emotional problem, and the supporter will give the appropriate emotional support and advice.

Your task is to act as an impartial scoring judge and rate the last response of supporter in the dialog in terms of the following aspects in order to evaluate the quality of the response. Please read and fully understand the following scoring criteria.

Note that please give the scores in the specified format, just the serial number and the relevant dimension score from the list of questions, without repeating the question itself. Also, do not add other extraneous prefixes and control characters.

## Evaluation Criteria:

(1) Empathy: Focusing on the comprehension of user emotions and the delineation of the underlying logical framework of user emotions.

### Options:

4 points: The system exhibits a high degree of anthropomorphism, going so far as to console users in a friendly manner and assist them in analyzing the underlying logic of emotions.
3 points: Providing emotional comfort during conversations and assisting users in analyzing the underlying logical framework of their emotions.
2 points: The lack of understanding of user emotions or the absence of mechanisms to analyze user emotions are the main factors.
1 point: The lack of understanding of user emotions and the absence of mechanisms to analyze user emotions are the main factors.
0 points: The disregard for user concerns, the absence of assistance in analyzing user issues, and even the imposition of negative effects on user emotions.

(2) Information: Focusing on Evaluating the Reasonableness and Quantity of Recommendations Provided by Emotion Assistants.

### Options:

4 points: There are many suggestions, and all of them are effective.
3 points: There are more than five suggestions, but some of them are ineffective. There are fewer than five suggestions, but all of them are very effective.
2 points: The suggestions are fewer than five, and some suggestions are effective, while others provide numerous suggestions, but none of them touch the root of the problem.
1 point: Have suggestions but ineffective, as well as no suggestions.
0 points: Suggestions were provided, but all of them were ineffective, and some even gave advice that could potentially harm the user.

(3) Humanoid: Focus on the differences between emotional assistants and humans.

### Options:

4 points: There is no apparent difference from human friends.
3 points: 1-2 traces can reveal that the AI assistant is a language model.
2 points: More than two traces can reveal that the AI assistant is a language model.
1 point: Structured responses, or responses in the form of ’As a large language model’ or robot-like replies.
0 points: The dialogue exhibits rigidity and lacks comprehension in terms of internalizing the content.

(4) Strategies: Evaluating the Accuracy and Appropriateness of Emotional Support Strategies Used by Assistants

### Options:

4 points: The strategies are numerous, well-tailored to the user's emotional state, and demonstrate high empathy and effectiveness in addressing the user's concerns.
3 points: More than five strategies are provided, but some lack empathy or relevance. Alternatively, fewer than five strategies are shared, but they are highly empathetic and directly address the user's core emotional needs.
2 points: Fewer than five strategies are provided, and they are a mix of relevant and irrelevant approaches. Alternatively, a large number of strategies are given, but they fail to address the user's emotional root issues.
1 point: Strategies are present but lack empathy or relevance. Some may appear dismissive or insufficiently supportive in the context of the user's concerns.
0 points: Strategies are counterproductive, exacerbating the user's distress or dismissing their concerns. Some suggestions may inadvertently harm the user's emotional well-being.

## Assessment Steps:

1. Read the conversation carefully to identify major topics and key points.
2. Read the Evaluation Criteria and compare them to the content of the conversation.
3. Based on the Evaluation Criteria, rate each aspect on a scale of 0 to 4, with 0 being the lowest and 4 being the highest.

What you need to do to evaluate this document:
{chat_history}

Please follow the response format below strictly, avoiding any positional bias and not letting the length of your response affect your evaluation. Evaluate the areas as objectively as possible.

## Answer format:

<Question number>: <Score>
'''
ROUND_QUESTION_NUM = 4
ROUND_WEIGHT = [0.1, 0.1, 0.1, 0.7]


FIX_PROMPT = '''
Here are the wrong responses and the reasons for their errors, so please learn from them and don't repeat them:
{bad_list}
'''


def build_user_prompt(chat_history, description, scene):
    prompt = USER_SYS_PROMPT.format(situation=description, emotion=scene)
    return prompt, USER_QUERY.format(chat_history=history_to_str(chat_history))
    

def build_supporter_prompt(chat_history, strategy):
    sys = SUPPORTER_SYS_PROMPT.format(character=CHARA_DETAIL)
    prompt = SUPPORTER_QUERY.format(
        strategy=strategy,
        strategy_detail=STRATEGY_MAP[strategy],
        chat_history=history_to_str(chat_history),
    )
    return sys, prompt

def build_strategy_prompt(chat_history):
    stratrgy_str = ''
    for k, v in STRATEGY_MAP.items():
        stratrgy_str += f'  - {k}: {v}\n'
    return STRATEGY_SYS_PROMPT, STRATEGY_QUERY.format(chat_history=history_to_str(chat_history, strategy=True), stratrgy_str=stratrgy_str)

    
@retry()
def gen_strategy(history, temperature=5):
    sys, query = build_strategy_prompt(history)
    output, in_token, out_token = generate(query, sys)

    match = re.findall(r'```.+?```', output, re.DOTALL)
    if len(match) == 1:
        output = match[0].removeprefix('```').removeprefix('json').removesuffix('```')
    else:
        match = re.findall(r'`.+?`', output, re.DOTALL)
        output = match[0].removeprefix('`').removeprefix('json').removesuffix('`')
    output = json.loads(output)

    score_list = [{"strategy": STRATEGY_KEY_MAP[k.lower()], "score": int(v)} for k, v in output.items()]

    exp_list = [math.exp(score['score']/temperature) for score in score_list]
    sum_exp_values = sum(exp_list)
    for i, score in enumerate(score_list):
        score['score'] = exp_list[i] / sum_exp_values
    return score_list

@retry()
def eval_round(history, strategy=None, assistant=None):
    query = EVAL_ROUND_PROMPT.format(chat_history=history_to_str(history + ([{"role": "supporter", "content": assistant, "strategy": strategy}] if assistant else []), strategy=True))
    output = generate(query)

    score = 0
    for i in range(ROUND_QUESTION_NUM):
        pattern = rf'\(?{i + 1}\)?:\s*(\d+)'
        matches = re.search(pattern, output, re.I)
        if not matches:
            print(repr(output))
            raise ValueError('bad eval round')
        score += int(matches.group(1)) * ROUND_WEIGHT[i]
    return score

@retry()
def gen_assistant(history, strategy):
    sys, query = build_supporter_prompt(history, strategy)
    output = generate(query, sys)

    return output.removeprefix('supporter').removeprefix('Supporter').strip("\"\': ")


@retry()
def gen_user(history, description, scene) -> tuple[str, int]:
    sys, query = build_user_prompt(history, description, scene)
    output = generate(query, sys)

    output = output.removeprefix('seeker').removeprefix('Seeker').strip("\"\': ")
    end = False
    if any(s in output.lower() for s in ['</end/>', '/end/', '<end>']):
        output = None
        end = True
    return output, end

@retry()
def eval_all(history, sim_round):
    score_list = []
    history = history[1:]
    for i in range(len(history) - sim_round * 2 - 1, len(history), 2):
        # print(history[:i])
        score_list.append(eval_round(history[:i]))
    return sum(score_list)/len(score_list)


C = 1
RW_BIAS = -3

def call_mcts(json_data, save_name):
    mcts = MCTS(
        init_assistant=SUPPORTER_START,
        c=C,
        sim_max_round=4,
        gen_strategy_fn=gen_strategy,
        gen_assistant_fn=gen_assistant,
        gen_user_fn=gen_user,
        eval_all_fn=eval_all,
        rw_bias=RW_BIAS
    )
    mcts.run(
        description = json_data['description'],
        scene = json_data['scene'],
        min_iter=100,
        min_end=1,
        max_iter=200,
        max_end=25,
        tmp_path=f'tmp/{save_name}',
        json_data=json_data
    )
    mcts.save(f'output/tree/{save_name}.pkl')
    mcts.draw(f'pic/{save_name}')

if __name__ == '__main__':

    with open('path_to_your_golden_data', encoding='utf-8') as f:
        data_list = json.load(f)

    start = 0
    end = 100

    data_list = data_list[start:end]

    threads = 10

    save_root = f'path_to_your_tree_folder'

    if threads <= 1:
        for situation, i in zip(data_list, range(start, end)):
            call_mcts(situation, f'{save_root}/{start}-{end}/{i}')
    else:
        with ThreadPoolExecutor(max_workers=threads) as executor:
            executor.map(call_mcts, data_list, [f'{save_root}/{start}-{end}/{i}' for i in range(start, end)])


