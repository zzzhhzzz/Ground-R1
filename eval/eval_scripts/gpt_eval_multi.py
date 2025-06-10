BASE_PROMPT = """
You are responsible for proofreading the answers, you need to give a score to the model's answer by referring to the standard answer, based on the given question. The full score is 1 point and the minimum score is 0 points. Please output the score in the form "score: <score>". The evaluation criteria require that the closer the model's answer is to the standard answer, the higher the score.
"""

PROMPT = """
question: %s
standard answer: %s
model's answer: %s
"""

import json
from tqdm import tqdm
from openai import OpenAI
import re
import time
import argparse
from collections import defaultdict
from multiprocessing import Pool

def parse_args():
    parser = argparse.ArgumentParser(description="Script with API key and API base URL as arguments")
    parser.add_argument("--api_key", type=str, required=True, help="API key for authentication")
    parser.add_argument("--api_base", type=str, required=True, help="Base URL for API access")
    parser.add_argument("--gpt_model", type=str, required=True, help="GPT Model")
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--eval_class", type=str, required=True)
    return parser.parse_args()

args = parse_args()
api_key = args.api_key
api_base = args.api_base
gpt_model = args.gpt_model

client = OpenAI(api_key=api_key, base_url=api_base)

def extract_answer_videor1(text):
    pattern = r'<answer>\s*(.*?)\s*</answer>'
    pattern2 = r'<answer>(.*)'
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    else:
        match2 = re.search(pattern2, text, re.DOTALL)
        if match2:
            return match2.group(1).strip()
        else:
            # return ""
            return text.strip()

def make_request_openai(content, extra_args={}):

    retry_times = 3
    while retry_times > 0:
        try:
            message = [{"role":"system","content": BASE_PROMPT}, {"role": "user", "content":content}]
            completion = client.chat.completions.create(
                model=gpt_model,
                messages=message,
            )
            response= completion.choices[0].message.content
            return response
        except Exception as e:
            print(e)
            time.sleep(1)
        finally:
            retry_times -= 1
    return 'unknown'


def get_score(question_text, gt_answer_text, pred_answer_text):
    content = PROMPT % (question_text, gt_answer_text, pred_answer_text)
    ret = make_request_openai(content)
    ret = ret.lower()
    if 'score' not in ret:
        return 0.0
    res = re.findall(r'score: ([\d\.]+)', ret)
    if len(res) != 1:
        return 0.0
    res = float(res[0])
    if res > 1.0:
        res = 1
    if res < 0.0:
        res = 0
    return res


input_file = args.input_file
output_file = args.output_file
eval_class = args.eval_class

def process_line(line):
    data = json.loads(line)
    
    dataset = data['dataset']

    if eval_class == "baseline":
        content = data['baseline_answer']
    elif eval_class == "stage2":
        content = data['model_answer_stage2']
    elif eval_class == "stage1":
        content = data['model_answer_stage1']
    elif eval_class == "0526":
        content = data['generation_list'][-1]

    ground_truth = data['solution']
    problem = data['problem']

    try:
        sol_match = re.search(r'<answer>(.*?)</answer>', ground_truth)
        ground_truth = sol_match.group(1).strip() if sol_match else ground_truth.strip()
    except:
        ground_truth = str(ground_truth)  # Clevr int

    student_answer = extract_answer_videor1(content)
    reward = get_score(problem, ground_truth, student_answer)

    data[f'{eval_class}_reward'] = reward
    # data['student_answer'] = student_answer

    return dataset, reward

def main():
    with open(input_file, "r", encoding="utf-8") as infile:
        lines = infile.readlines()

    correct_counts = defaultdict(int)
    total_counts = defaultdict(int)

    with Pool(processes=10) as pool:
        for dataset, reward in tqdm(pool.imap(process_line, lines), total=len(lines), desc="Processing"):
            total_counts[dataset] += 1
            correct_counts[dataset] += reward

    with open(output_file, "w", encoding="utf-8") as outfile:
        for dataset in total_counts:
            accuracy = correct_counts[dataset] / total_counts[dataset] if total_counts[dataset] > 0 else 0
            result_line = f"Dataset: {dataset}, Accuracy: {accuracy:.4f} ({correct_counts[dataset]}/{total_counts[dataset]})"
            print(result_line)
            outfile.write(result_line + "\n")

if __name__ == "__main__":
    main()
