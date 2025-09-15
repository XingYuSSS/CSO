import os
import json


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

strategy_list = ''
for k, v in STRATEGY_MAP.items():
    strategy_list += f'  - {k}: {v}\n\n'
strategy_list = strategy_list[:-2]

SYSTEM = '''
You are an emotional supporter. You are talking to the seeker. Your should response based on the given strategies.

The strategy should be chosen from the following {strategy_num} types of strategy:

{strategy_list}

You should response in the following format:

(<strategy>) <response>
'''.format(strategy_num=len(STRATEGY_MAP), strategy_list=strategy_list)



in_file = ''
out_file =''

with open(in_file, 'r', encoding='utf-8') as f:
    data = json.load(f)

change_data = []

for d in data:
    for i in range(1, len(d['messages']), 2):
        message = d['messages'][:i]
        message = [{
            "role": m['role'],
            "content": m['content'] if m['role'] == 'user' else f"({m['strategy']}) {m['content']}",
        } for m in message]
        message.insert(0, {
            'role': 'system',
            'content': SYSTEM
        })
        change_data.append({
            "scene": d['scene'],
            "description": d['description'],
            "messages": message + [{
                "role": 'assistant',
                "content": f"({d['messages'][i]['strategy']}) {d['messages'][i]['content']}",
            }],
            "label": True
        })
        for n in d['messages'][i]['negative']:
            change_data.append({
                "scene": d['scene'],
                "description": d['description'],
                "messages": message + [{
                    "role": 'assistant',
                    "content": f"({n['strategy']}) {n['content']}",
                }],
                "label": False
            })

print(len(change_data))

seen = set()
unique_dict_list = []

for d in change_data:
    serialized_d = json.dumps(d, sort_keys=True)
    if serialized_d not in seen:
        seen.add(serialized_d)
        unique_dict_list.append(d)

print(len(unique_dict_list))

with open(out_file, 'w', encoding='utf-8') as f:
    json.dump(unique_dict_list, f, ensure_ascii=False, indent=2)

