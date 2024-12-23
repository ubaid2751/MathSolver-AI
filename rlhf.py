from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from colorama import Fore, Style
from main import PAL_prompt
import ollama
from typing import Dict, Generator

MODEL_NAME = "llama3.2:1b"

def extract_code(response: str):
    start, end = -1, -1
    if "<code>" in response and "</code>" in response:
        start = response.find("<code>") + len("<code>")
        end = response.find("</code>", start)
    elif "```python" in response and "```" in response:
        start = response.find("```python") + len("```python")
        end = response.find("```", start)

    if start != -1 and end != -1:
        return response[start:end].strip()
    return "<NO CODE FOUND>"

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
    try:
        ollama_client = ollama.Client()
        response = ollama_client.chat(
            model=MODEL_NAME,
            messages=messages,
            stream=True,
            options={"temperature": 0.0}
        )
        for chunk in response:
            if "message" in chunk and "content" in chunk["message"]:
                yield chunk
            else:
                raise KeyError("Unexpected response format from Ollama API.")
    except Exception as e:
        print(f"{Fore.RED}Error in generate_response: {e}{Style.RESET_ALL}")
        yield {"message": {"content": "<ERROR: Unable to generate response>"}}


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

feedback_history = []

# Collect feedback from the user
def collect_feedback(question: str, generated_answer: str):
    print(f"{Fore.MAGENTA}Question: {Style.RESET_ALL}{question}")
    print(f"{Fore.BLUE}Generated Answer: {Style.RESET_ALL}{generated_answer}")
    feedback = input(f"{Style.BRIGHT}{Fore.YELLOW}Is response GOOD or BAD: {Style.RESET_ALL}")
    try:
        if feedback.upper() == "GOOD":
            feedback = 1
        elif feedback.upper() == "BAD":
            feedback = -1
        else:
            feedback = 0
        feedback_history.append({
            "question": question,
            "generated_answer": generated_answer,
            "feedback": feedback
        })
        return feedback
    except Exception as e:
        print(f"{Fore.RED}Error while collecting feedback: {e}{Style.RESET_ALL}")
        return 0

# Initialize model
vectorizer = TfidfVectorizer()
classifier = LogisticRegression()

# Train the model
def train_model(feedback_history):
    if not feedback_history:
        print(f"{Fore.RED}No feedback data available for training.{Style.RESET_ALL}")
        return

    questions = [entry["question"] for entry in feedback_history]
    labels = [1 if entry["feedback"] > 0 else 0 for entry in feedback_history]

    if not questions or not labels:
        print(f"{Fore.RED}Insufficient data to train the model.{Style.RESET_ALL}")
        return

    X = vectorizer.fit_transform(questions)
    y = labels

    classifier.fit(X, y)
    print(f"{Fore.GREEN}Model trained with {len(questions)} samples.{Style.RESET_ALL}")


# Predict question quality
def is_good_question(question: str) -> bool:
    X = vectorizer.transform([question])
    prediction = classifier.predict(X)
    print(f"{Fore.CYAN}Prediction for question quality: {prediction[0]}{Style.RESET_ALL}")
    return prediction[0] == 1

# Refine the prompt
def refine_prompt(question: str, current_prompt: str) -> str:
    if is_good_question(question):
        print(f"{Fore.GREEN}The question seems good. Keeping the current strategy.{Style.RESET_ALL}")
        return current_prompt
    else:
        print(f"{Fore.RED}The question is not good. Refining the prompt.{Style.RESET_ALL}")
        return current_prompt + "\nEnsure clarity and concise language. Focus on numerical calculations."

# Main workflow with learning
def worker_with_feedback_learning(question: str, n_steps: int = 5):
    global ITER
    ITER = 0
    current_prompt = PAL_prompt.replace("{problem}", question)

    while ITER < n_steps:
        response = user_question(question)
        result = execute_code(response)

        if result != "<NO RESULT FOUND>" and result != "ERROR":
            print(f"{Style.BRIGHT}{Fore.GREEN}<ANSWER FOUND>{Style.RESET_ALL}")
            print(f"{Style.BRIGHT}{Fore.MAGENTA}Question:{Style.RESET_ALL} {question}")
            print(f"{Style.BRIGHT}{Fore.MAGENTA}Answer:{Style.RESET_ALL} {result}")

            feedback = collect_feedback(question, result)
            if feedback == 0:
                print(f"{Fore.YELLOW}No feedback provided. Ending loop.{Style.RESET_ALL}")
                break

            if feedback < 0:
                current_prompt = refine_prompt(question, current_prompt)

            train_model(feedback_history)
        else:
            ITER += 1
            if ITER >= n_steps:
                print(f"{Style.BRIGHT}{Fore.RED}<NO ANSWER FOUND>{Style.RESET_ALL}")
                break
            else:
                print(f"{Style.BRIGHT}{Fore.YELLOW}RESTARTING...{Style.RESET_ALL}")

# Example usage

ques = "Roger has 5 tennis balls. He buys 2 more cans of tennis balls. Each can has 3 tennis balls. How many tennis balls does he have now?"

while True:
    ques = input(f"{Style.BRIGHT}{Fore.MAGENTA}Enter your question or type \\exit: {Style.RESET_ALL}")
    if ques == r"\exit":
        print(f"{Style.BRIGHT}{Fore.GREEN}Thank you!!{Style.RESET_ALL}")
        break
    worker_with_feedback_learning(ques)