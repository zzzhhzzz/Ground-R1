import re
from math_verify import ExprExtractionConfig, LatexExtractionConfig, StringExtractionConfig, parse, verify
from rouge_score import rouge_scorer
import os

choices = ["a", "b", "c", "d"]


def compute_iou(gt_bbox, student_bbox):

    x1_gt, y1_gt, x2_gt, y2_gt = gt_bbox
    x1_st, y1_st, x2_st, y2_st = student_bbox

    x1_inter = max(x1_gt, x1_st)
    y1_inter = max(y1_gt, y1_st)
    x2_inter = min(x2_gt, x2_st)
    y2_inter = min(y2_gt, y2_st)

    inter_width = max(0, x2_inter - x1_inter)
    inter_height = max(0, y2_inter - y1_inter)
    inter_area = inter_width * inter_height

    gt_area = (x2_gt - x1_gt) * (y2_gt - y1_gt)
    student_area = (x2_st - x1_st) * (y2_st - y1_st)

    union_area = gt_area + student_area - inter_area

    iou = inter_area / union_area if union_area > 0 else 0

    iou = iou

    return iou


def compute_giou(gt_bbox, student_bbox):

    x1_gt, y1_gt, x2_gt, y2_gt = gt_bbox
    x1_st, y1_st, x2_st, y2_st = student_bbox

    x1_inter = max(x1_gt, x1_st)
    y1_inter = max(y1_gt, y1_st)
    x2_inter = min(x2_gt, x2_st)
    y2_inter = min(y2_gt, y2_st)

    inter_width = max(0, x2_inter - x1_inter)
    inter_height = max(0, y2_inter - y1_inter)
    inter_area = inter_width * inter_height

    gt_area = (x2_gt - x1_gt) * (y2_gt - y1_gt)
    student_area = (x2_st - x1_st) * (y2_st - y1_st)

    union_area = gt_area + student_area - inter_area

    iou = inter_area / union_area if union_area > 0 else 0

    x1_c = min(x1_gt, x1_st)
    y1_c = min(y1_gt, y1_st)
    x2_c = max(x2_gt, x2_st)
    y2_c = max(y2_gt, y2_st)

    c_area = (x2_c - x1_c) * (y2_c - y1_c)

    giou = iou - (c_area - union_area) / c_area if c_area > 0 else iou

    giou_scaled = giou
    return giou_scaled

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
    parser.add_argument("--output_file", type=str, default="")

    return parser.parse_args()

args = parse_args()
input_file = args.input_file
output_file = args.output_file
write_output = bool(output_file.strip())

dataset_rewards_stage1 = defaultdict(float)
dataset_rewards_stage2 = defaultdict(float)
dataset_iou_stage1 = defaultdict(list)
dataset_iou_stage2 = defaultdict(list)

with open(input_file, "r", encoding="utf-8") as infile, \
     (open(output_file, "w", encoding="utf-8") if write_output else open(os.devnull, 'w')) as outfile:
    
    for line in infile:
        data = json.loads(line)
        dataset = data['dataset']
        content1 = data['model_answer_stage1']
        content2 = data['model_answer_stage2']
        ground_truth = data['solution']
        bbox_stage1 = data['bbox_to_cal_iou_score_stage1']
        bbox_stage2 = data['bbox_to_cal_iou_score_stage1']
        bbox_gt = data["bboxs"][0]

        try:
            sol_match = re.search(r'<answer>(.*?)</answer>', ground_truth)
            ground_truth = sol_match.group(1).strip() if sol_match else ground_truth.strip()
        except:
            ground_truth = str(ground_truth)

        def get_reward(content):
            if dataset in ["v7w", "gqa", "openimages"]:
                return v7w(content, ground_truth)
            elif dataset == "CLEVR":
                return Math_eureka(content, ground_truth)
            elif dataset in ["docvqa", "infographicsvqa"]:
                return ocr_and_freeform(content, ground_truth, dataset)
            elif dataset in ["textcap", "textvqa", "dude", "sroie", "flickr30k"]:
                return ocr_and_freeform(content, ground_truth, dataset)
            elif dataset in ["vsr", "cub"]:
                return openend_datasets(content, ground_truth)
            else:
                return 0, ""

        reward1, student_answer1 = get_reward(content1)
        reward2, student_answer2 = get_reward(content2)
        
        try:
            iou_stage1 = compute_iou(bbox_gt, bbox_stage1)
        except Exception as e:
            iou_stage1 = 0.0
        
        try:
            iou_stage2 = compute_iou(bbox_gt, bbox_stage2)
        except Exception as e:
            iou_stage2 = 0.0



        if reward2 > reward1:
            data['reward_stage1'] = reward1
            data['reward_stage2'] = reward2
            data['student_answer_stage1'] = student_answer1
            data['student_answer_stage2'] = student_answer2
            outfile.write(json.dumps(data, ensure_ascii=False) + "\n")

        dataset_rewards_stage1[dataset] += reward1
        dataset_rewards_stage2[dataset] += reward2
        dataset_iou_stage1[dataset].append(iou_stage1)
        dataset_iou_stage2[dataset].append(iou_stage2)

print("==== Stage 1 Results ====")
for dataset, total_reward in dataset_rewards_stage1.items():
    ious = dataset_iou_stage1[dataset]
    avg_iou = sum(ious) / len(ious) if ious else 0
    print(f"Dataset: {dataset}, Total Reward (Stage 1): {total_reward:.2f}, Average IOU (Stage 1): {avg_iou:.2f}")

print("\n==== Stage 2 Results ====")
for dataset, total_reward in dataset_rewards_stage2.items():
    ious = dataset_iou_stage2[dataset]
    avg_iou = sum(ious) / len(ious) if ious else 0
    print(f"Dataset: {dataset}, Total Reward (Stage 2): {total_reward:.2f}, Average IOU (Stage 2): {avg_iou:.2f}")