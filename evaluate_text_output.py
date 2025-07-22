import json
import re
import csv

# 4. 给输入文件增加变量名
input_filename = 'output/medqa_advanced_63samples.json'
output_filename = 'evaluation/medqa_advanced_63samples.csv'

# 读取原始 JSON 文件
with open(input_filename, 'r', encoding='utf-8') as f:
    data = json.load(f)

def extract_final_answer_or_answer(text):
    match_final = re.search(r'(The final answer.*?[\.\n])', text, re.IGNORECASE)
    if match_final:
        text = match_final.group(1).strip()
    else:
        match_answer = re.search(r'(\n\n\*\*Answer:\*\* *\([A-E]\).*)', text)
        if match_answer:
            text = match_answer.group(1).strip()
        else:
            words = text.strip().split()
            last15 = ' '.join(words[-15:]) if len(words) >= 15 else ' '.join(words)
            match_correct = re.search(r'(The correct answer is \*\*\([A-E]\)\*\*)', last15)
            if match_correct:
                text = match_correct.group(1).strip()
            else:
                match_star = re.search(r'(\*\*\([A-E]\)\*\*)', last15)
                if match_star:
                    text = match_star.group(1).strip()
                else:
                    match_diag = re.search(r'(\*\*The most likely diagnosis is \([A-E]\)\*\*)', last15)
                    if match_diag:
                        text = match_diag.group(1).strip()
                    else:
                        text = last15
    match_letter = re.search(r'\b([A-E])\b', text)
    if match_letter:
        return match_letter.group(1)
    else:
        return ''

processed_data = []
total_correct = 0
total_samples = len(data)

# Track correct/total for each difficulty
correct_by_diff = {'basic': 0, 'intermediate': 0, 'advanced': 0}
total_by_diff = {'basic': 0, 'intermediate': 0, 'advanced': 0}

for sample in data:
    question_id = sample.get('question_id', '')
    label = sample.get('label', '')
    resp = sample.get('response', '')

    response_text = ''
    if isinstance(resp, dict):
        if 'final_answer' in resp:
            response_text = resp['final_answer']
        else:
            response_text = resp.get('0.0', '')
            if not response_text and 'majority_vote' in resp:
                response_text = resp['majority_vote']
    else:
        response_text = str(resp)
    
    response = extract_final_answer_or_answer(response_text)

    difficulty = sample.get('difficulty', '')
    token_usage = sample.get('token_usage', {})
    input_tokens = token_usage.get('input_tokens', '')
    output_tokens = token_usage.get('output_tokens', '')
    total_tokens = token_usage.get('total_tokens', '')
    
    is_correct = 1 if label == response else 0
    total_correct += is_correct

    # Track per-difficulty
    if difficulty in correct_by_diff:
        correct_by_diff[difficulty] += is_correct
        total_by_diff[difficulty] += 1

    processed_data.append([
        question_id, label, response, difficulty, 
        input_tokens, output_tokens, total_tokens, is_correct
    ])

# 2. 增加计算准确度的代码
accuracy = total_correct / total_samples if total_samples > 0 else 0

# Report per-difficulty accuracy
for diff in ['basic', 'intermediate', 'advanced']:
    total = total_by_diff[diff]
    correct = correct_by_diff[diff]
    acc = correct / total if total > 0 else 0
    print(f'Accuracy for {diff}: {acc:.4f} ({correct}/{total})')

# 1. 输出文件格式改成csv
with open(output_filename, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    header = ["question_id", "label", "response", "difficulty", "input_tokens", "output_tokens", "total_tokens", "correct"]
    writer.writerow(header)
    writer.writerows(processed_data)

# 3. 打印 准确度
print(f'Accuracy: {accuracy:.4f}')
print(f'CSV导出完成：{output_filename}')

# 英文注释：This script cleans the response field, calculates accuracy, and exports to a CSV file.
# 中文注释：本脚本清洗response字段，计算准确率，并导出为CSV文件。 