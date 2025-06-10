import re
from math_verify import ExprExtractionConfig, LatexExtractionConfig, StringExtractionConfig, parse, verify
from rouge_score import rouge_scorer

choices = ["a", "b", "c", "d"]


def extract_answer_with_tags(text):
    match = re.search(r'<answer>\s*(.*?)\s*</answer>', text)
    if match:
        return match.group(1)
    return None

# For multi-choice question     ## v7w
def v7w(content, ground_truth):

    reward = 0.0
    try:
        content_match = re.search(r'<answer>\s*([A-Da-d])', content, re.DOTALL)
        student_answer = content_match.group(1).strip() if content_match else content.strip()
        
        # Compare the extracted answers
        if student_answer.lower() == ground_truth.lower():
            reward = 1.0
        else:
            content_match2 = re.search(r"\b([A-D])\.", content, re.DOTALL)
            student_answer = content_match2.group(1).strip() if content_match2 else content.strip()
            if student_answer.lower() == ground_truth.lower():
                reward = 1.0
            else:
                student_answer = extract_answer_videor1(content)
                if student_answer.lower() == ground_truth.lower():
                    reward = 1.0
    
    except Exception as e:
        print(f"accuracy_reward_cal_Error:{e}")
        student_answer = f"accuracy_reward_cal_Error:{e}"
        pass
    
    return reward, student_answer

# For many possible answer question     ## docVQA InfographicsVQA
def docVQA(content, ground_truth):

    reward = 0.0
    try:
        match = re.search(r'<answer>(.*?)</answer>', content, re.DOTALL)
        student_answer = match.group(1).strip() if match else content.strip()
        if student_answer.lower() in ground_truth:
            reward = 1.0

    except Exception as e:
        print(f"accuracy_reward_cal_Error: {e}")
        student_answer = f"accuracy_reward_cal_Error:{e}"
        pass

    return reward, student_answer

def Math_eureka(content, ground_truth):
    
    reward = 0.0
    response = extract_answer_with_tags(content)
    if response != None:
        response = response
    else:
        try:
            response = content.split("<answer>")[-1]
        except:
            response = content.split("\n")[-1]

    content= response
    answer_parsed = content
    gold_parsed = parse(ground_truth)
    if len(gold_parsed) != 0:
        answer_parsed = parse(
            content,
            extraction_config=[StringExtractionConfig(), LatexExtractionConfig(), ExprExtractionConfig()],
        )
        try:
            reward = float(verify(answer_parsed, gold_parsed))
        except Exception:
            pass

        if reward == 0.0:
            try:
                content_match = re.search(r"<answer>(.*?)</answer>", response)
                student_answer = content_match.group(1).strip() if content_match else content.strip()
                student_answer = student_answer.replace("</answer>", "").replace("<answer>", "").strip()
                for answer in gold_parsed:
                    if str(answer).lower() in choices:
                        if str(answer).lower() in student_answer.lower():
                            choices_other = [choice for choice in choices if choice != str(answer).lower()]
                            if all(choice not in student_answer.lower() for choice in choices_other):
                                reward = 1.0
            except Exception:
                pass
    else:
        reward = 1.0
        print("Failed to parse gold solution: ", ground_truth)

    return reward, answer_parsed

## vsr
def openend_datasets(content, ground_truth): 

    reward = 0.0

    try:
        student_answer = extract_answer_videor1(content)
        if student_answer.lower() == ground_truth.lower():
            reward = 1.0
        elif ground_truth.lower() in student_answer.lower():
            reward = 0.6

    except Exception as e:
        print(f"accuracy_reward_cal_Error: {e}")
        student_answer = f"accuracy_reward_cal_Error:{e}"
        pass
    
    return reward, student_answer

## Ocr and freeform
'''
def extract_answer_videor1(text):
    pattern = r'<answer>\s*(.*?)\s*</answer>'
    pattern2 = r'<answer>(.*)'
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    else:
        return ""
'''
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

def ocr_and_freeform(content, ground_truth, dataset):

    reward = 0.0

    def wer(reference, hypothesis):
        ref_words = reference.split()
        hyp_words = hypothesis.split()
        m = len(ref_words)
        n = len(hyp_words)
        d = [[0]*(n+1) for _ in range(m+1)]
        for i in range(m+1):
            d[i][0] = i
        for j in range(n+1):
            d[0][j] = j
        for i in range(1, m+1):
            for j in range(1, n+1):
                if ref_words[i-1] == hyp_words[j-1]:
                    d[i][j] = d[i-1][j-1]
                else:
                    d[i][j] = 1 + min(d[i-1][j], d[i][j-1], d[i-1][j-1])
        return d[m][n] / max(1, m)


    def compute_rouge_score(reference, hypothesis, use_stemmer=True):
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=use_stemmer)
        scores = scorer.score(reference, hypothesis)
        average_fmeasure = (scores['rouge1'].fmeasure + scores['rouge2'].fmeasure + scores['rougeL'].fmeasure) / 3
        return average_fmeasure

    try:
        student_answer = extract_answer_videor1(content)
        if dataset == "docvqa" or dataset == "textcap" or dataset == "textvqa" or dataset == "sroie" or dataset == "dude" or dataset == "infographicsvqa":
            error_rate01 = wer(ground_truth.lower(), student_answer.lower())  ### ocr
            error_rate02 = wer(student_answer.lower(), ground_truth.lower())
            reward = 1 - min(error_rate01, error_rate02)
            reward = max(0.0, min(1.0, reward))
        elif dataset == "flickr30k":    ### freeform
            score = compute_rouge_score(ground_truth.lower(), student_answer.lower())
            reward = max(0.0, min(1.0, score))

    except Exception as e:
        print(f"accuracy_reward_cal_Error: {e}")
        student_answer = f"accuracy_reward_cal_Error:{e}"
        pass
    return reward, student_answer


import json
from collections import defaultdict

import argparse

def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True)

    return parser.parse_args()

args = parse_args()
input_file = args.input_file

dataset_rewards = defaultdict(float)

with open(input_file, "r", encoding="utf-8") as infile:
    for line in infile:
        data = json.loads(line)
        dataset = data['dataset']
        content = data['baseline_answer']
        ground_truth = data['solution']
        try:
            sol_match = re.search(r'<answer>(.*?)</answer>', ground_truth)
            ground_truth = sol_match.group(1).strip() if sol_match else ground_truth.strip()
        except:
            ground_truth = str(ground_truth) ## Clevr int
        if dataset == "v7w" or dataset == "gqa" or dataset == "openimages":
            reward, student_answer = v7w(content, ground_truth)
        elif dataset == "CLEVR":
            reward, student_answer = Math_eureka(content, ground_truth)
        elif dataset == "docvqa" or dataset == "infographicsvqa":
            reward, student_answer = ocr_and_freeform(content, ground_truth, dataset)
        elif dataset == "textcap" or dataset == "textvqa" or dataset == "dude" or dataset == "sroie" or dataset == "flickr30k":
            reward, student_answer = ocr_and_freeform(content, ground_truth, dataset)
        elif dataset == "vsr" or dataset == "cub":
            reward, student_answer = openend_datasets(content, ground_truth)
        else:
            reward = 0
            student_answer = ""

        data['reward'] = reward
        data['student_answer'] = student_answer
        dataset_rewards[dataset] += reward

for dataset, total_reward in dataset_rewards.items():
    print(f"Dataset: {dataset}, Total Reward: {total_reward}")
