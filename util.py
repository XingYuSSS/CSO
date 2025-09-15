import os
import json
import time
import functools

from openai import OpenAI
from dotenv import load_dotenv



load_dotenv()

openai = OpenAI(
    api_key=os.getenv('openai_key'),
    base_url=os.getenv('openai_url'),
)

def generate(query, system = None):
    global openai
    if openai.is_closed():
        openai = OpenAI(
        api_key=os.getenv('openai_key'),
        base_url=os.getenv('openai_url'),
    )
    msg = [{"role": "user", "content": query}]
    if system:
        msg.insert(0, {"role": "system", "content": system})
    chat_completion = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=msg,
        stream=False,
    )
    return chat_completion.choices[0].message.content

def retry(max_attempts=None, delay=1):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            attempts = 0
            while max_attempts is None or attempts < max_attempts:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    print('err:', e, '\n')
                    last_exception = e
                    attempts += 1
                    if delay > 0:
                        time.sleep(delay)
            raise last_exception
        return wrapper
    return decorator


def history_to_str(history, strategy=False) -> bool:
    chat_history = ''
    for h in history:
        chat_history += f'seeker: {h["content"]}\n' if h['role'] == 'user' else 'supporter: ' + (f'({h["strategy"]}) ' if strategy else '') + f'{h["content"]}\n'
    chat_history = chat_history[:-1]
    return chat_history

def as_bool(v: str | bool):
    if type(v) == bool:
        return v
    if v.lower() == 'true':
        return True
    if v.lower() == 'false':
        return False
    raise ValueError('not bool')


