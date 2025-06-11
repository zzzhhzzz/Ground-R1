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
            return text.strip()   ### For only answer no think format

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


def deepeyes_visual_toolbox(content, ground_truth):

    reward = 0.0
    def compute_rouge_score(reference, hypothesis, use_stemmer=True):
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=use_stemmer)
        scores = scorer.score(reference, hypothesis)
        average_fmeasure = (scores['rouge1'].fmeasure + scores['rouge2'].fmeasure + scores['rougeL'].fmeasure) / 3
        return average_fmeasure
    
    try:
        student_answer = extract_answer_videor1(content)
        if "yes" in ground_truth.split()[0].lower() and student_answer.split()[0].lower() == "yes":
            reward = 1.0
        elif "no" in ground_truth.split()[0].lower() and student_answer.split()[0].lower() == "no":
            reward = 1.0
        else:
            score = compute_rouge_score(ground_truth.lower(), student_answer.lower())
            reward = max(0.0, min(1.0, score))

    except Exception as e:
        print(f"accuracy_reward_cal_Error: {e}")
        student_answer = f"accuracy_reward_cal_Error:{e}"
        pass
    return reward, student_answer

import os
import json
from filelock import FileLock

def output_json_for_selection(log_path, problem_id, image, dataset, rewards):
    if log_path:
        data_select = {
            "problem_id": problem_id[0],
            "image": image[0],
            "dataset": dataset[0],
            "rewards": rewards,
            "rewards_sum": sum(rewards)
        }

        lock_path = log_path + ".lock"  # 创建锁文件
        lock = FileLock(lock_path)

        with lock:  # 进入锁定状态
            with open(log_path, "a", encoding="utf-8") as f:  # 追加模式，不覆盖
                f.write(json.dumps(data_select, ensure_ascii=False) + "\n")

# def bbox_reward_without_gt():
    # bbox != [], reward=1.0