import ollama
from typing import Dict, Generator
from colorama import Fore, Style, init

init(autoreset=True)

ITER = 0
MODEL_NAME = "llama3.2:1b"
PAL_prompt = """
You are a helpful assistant that solves math problems by writing Python programs.
You only respond with Python code blocks and you always place the answer in a variable called result.

Here's an example:

Problem: What is the result of the sum of the squares of 3 and 4?

Let's think step by step.

<code>
# Step 1: Define the numbers
a = 3
b = 4

# Step 2: Calculate the squares of the numbers
a_squared = a ** 2
b_squared = b ** 2

# Step 3: Calculate the sum of the squares
result = a_squared + b_squared

# Step 4: Return the result
result
</code>

Now solve this new problem using the same approach, remember you must place the answer in a variable named 'result'.

Problem: {problem} Let's think step by step."""

def extract_code(response: str):
    if response.find("<code>"):
        start = response.find("<code>") + len("<code>")
        end = response.find("</code>", start)
    if response.find("```python"):
        start = response.find("```python") + len("```python")
        end = response.find("```", start)
    code = response[start:end].strip()
    return code

def execute_code(response: str):
    try:
        code = extract_code(response)
        print(f"{Style.BRIGHT}{Fore.YELLOW}APPLYING...{Style.RESET_ALL}")
        local_env = {}
        _ = exec(code, {}, local_env)
        return local_env.get('result', "<NO RESULT FOUND>")
    except Exception as e:
        print(f"{Style.BRIGHT}{Fore.RED}<ERROR: {e}>{Style.RESET_ALL}")
        return "ERROR"

def generate_response(messages: Dict) -> Generator:
    ollama_client = ollama.Client()
    response = ollama_client.chat(
        model=MODEL_NAME,
        messages=messages,
        stream=True,
        options={
            "temperature": 0.5
        }
    )
    for chunk in response:
        yield chunk

def user_question(question: str):
    messages = [
        {
            "role": "user",
            "content": PAL_prompt.replace("{problem}", question)
        }
    ]
    ans = ""
    print(f"{Style.BRIGHT}{Fore.CYAN}THINKING...{Style.RESET_ALL}")
    for chunk in generate_response(messages):
        print(f"{Style.DIM}{chunk['message']['content']}", end='', flush=True)
        ans += chunk['message']['content']
    print(f"{Style.RESET_ALL}")
    return ans

def worker(question: str, n_steps: int = 5):
    global ITER
    ITER = 0
    while ITER < n_steps:
        response = user_question(question)
        result = execute_code(response)
        
        if result != "<NO RESULT FOUND>" and result != "ERROR":
            ITER = 0
            print(f"{Style.BRIGHT}{Fore.GREEN}<ANSWER FOUND>{Style.RESET_ALL}")
            print(f"{Style.BRIGHT}{Fore.MAGENTA}Question:{Style.RESET_ALL} {question}")
            print(f"{Style.BRIGHT}{Fore.MAGENTA}Answer:{Style.RESET_ALL} {result}")
            return
        else:
            ITER += 1
            if ITER >= n_steps:
                print(f"{Style.BRIGHT}{Fore.RED}<NO ANSWER FOUND>{Style.RESET_ALL}")
                return
            else:
                print(f"{Style.BRIGHT}{Fore.YELLOW}RESTARTING...{Style.RESET_ALL}")

ques = "Roger has 5 tennis balls. He buys 2 more cans of tennis balls. Each can has 3 tennis balls. How many tennis balls does he have now?"

while True:
    ques = input(f"{Style.BRIGHT}{Fore.MAGENTA}Enter your question or type \\exit: {Style.RESET_ALL}")
    if ques == r"\exit":
        print(f"{Style.BRIGHT}{Fore.GREEN}Thank you!!{Style.RESET_ALL}")
        break
    worker(ques)