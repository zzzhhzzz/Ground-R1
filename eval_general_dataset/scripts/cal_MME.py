import os
import argparse
import json
from collections import defaultdict
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
import re
parser = argparse.ArgumentParser()
parser.add_argument('--results_file',type=str, required=True)
args = parser.parse_args()

eval_type_dict = {
    "Perception": ["existence", "count", "position", "color", "posters", "celebrity", "scene", "landmark", "artwork", "OCR"],
    "Cognition": ["commonsense_reasoning", "numerical_calculation", "text_translation", "code_reasoning"]
}


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

class CalculateMetrics:
    def parse_pred_ans(self, pred_ans):
        # pred_ans = pred_ans.strip().lower()
        pred_ans = extract_answer_videor1(pred_ans).lower()
        if pred_ans in ["yes", "no"]:
            return pred_ans
        prefix_pred_ans = pred_ans[:4]
        if "yes" in prefix_pred_ans:
            return "yes"
        elif "no" in prefix_pred_ans:
            return "no"
        else:
            return "other"

    def compute_metric(self, gts, preds):
        assert len(gts) == len(preds)

        label_map = {"yes": 1, "no": 0, "other": -1}
        gts = [label_map[x] for x in gts]
        preds = [label_map[x] for x in preds]

        acc = accuracy_score(gts, preds)

        clean_gts = []
        clean_preds = []
        other_num = 0
        for gt, pred in zip(gts, preds):
            if pred == -1:
                other_num += 1
                continue
            clean_gts.append(gt)
            clean_preds.append(pred)

        if clean_preds:
            precision = precision_score(clean_gts, clean_preds)
            recall = recall_score(clean_gts, clean_preds)
            conf_mat = confusion_matrix(clean_gts, clean_preds, labels=[1, 0]).tolist()
        else:
            precision = 0.0
            recall = 0.0
            conf_mat = [[0, 0], [0, 0]]

        return {
            "acc": acc,
            "precision": precision,
            "recall": recall,
            "TP": conf_mat[0][0],
            "FN": conf_mat[0][1],
            "TN": conf_mat[1][1],
            "FP": conf_mat[1][0],
            "other_num": other_num
        }

    def process_results(self, results_file):
        # 读取所有数据
        data = []
        with open(results_file, 'r') as f:
            for line in f:
                data.append(json.loads(line.strip()))

        # 按任务分类
        task_data = defaultdict(list)
        for item in data:
            task_name = item["dataset"]
            task_data[task_name].append(item)

        scores = 0
        task_score_dict = {}

        for task_name, items in task_data.items():
            # 按图片分组
            image_group = defaultdict(list)
            for item in items:
                image_group[item["Image"]].append(item)

            img_num = len(image_group)
            task_other_ans_num = 0
            task_score = 0
            acc_plus_correct_num = 0
            gts = []
            preds = []

            for img_items in image_group.values():
                img_correct_num = 0

                for item in img_items:
                    gt_ans = item["solution"].strip().lower()

                    if "stage1_answer" in item:
                        pred_ans = item["stage1_answer"].strip().lower()
                    elif "model_answer_stage2" in item:
                        pred_ans = item["model_answer_stage2"].strip().lower()

                    assert gt_ans in ["yes", "no"], f"Unexpected gt_ans: {gt_ans}"

                    pred_ans = self.parse_pred_ans(pred_ans)
                    assert pred_ans in ["yes", "no", "other"], f"Unexpected pred_ans: {pred_ans}"

                    gts.append(gt_ans)
                    preds.append(pred_ans)

                    if gt_ans == pred_ans:
                        img_correct_num += 1

                    if pred_ans == "other":
                        task_other_ans_num += 1

                # acc_plus: 一张图片的所有问题都答对
                if img_correct_num == len(img_items):
                    acc_plus_correct_num += 1

            # 计算指标
            metric_dict = self.compute_metric(gts, preds)
            acc_plus = acc_plus_correct_num / img_num if img_num > 0 else 0.0
            metric_dict["acc_plus"] = acc_plus

            for k, v in metric_dict.items():
                if k in ["acc", "acc_plus"]:
                    task_score += v * 100

            task_score_dict[task_name] = task_score
            scores += task_score

        # 打印总分
        print("Total Score:", scores, "\n")
        for task_name, score in task_score_dict.items():
            print(f"\t{task_name} score: {score}")
        print()

if __name__ == "__main__":
    metric = CalculateMetrics()
    metric.process_results(args.results_file)
