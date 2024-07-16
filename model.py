import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# 모델과 토크나이저 로드
model_path = "bert_fine_tuned.pt" ## saved in Modeling.ipynb
tokenizer = AutoTokenizer.from_pretrained("jhgan/ko-sroberta-multitask")
model = AutoModelForSequenceClassification.from_pretrained("jhgan/ko-sroberta-multitask", num_labels=4)
model.load_state_dict(torch.load(model_path))

# 레이블을 정수로 변환
label_map = {
    0: "normal",
    1: "성범죄",
    2: "마약",
    3: "해킹"
}

def classify_sentence(sentence, model, tokenizer):
    # 문장 토큰화 및 인코딩
    inputs = tokenizer(sentence, return_tensors='pt', padding=True, truncation=True)
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']

    # 모델 예측
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits

    # 소프트맥스를 적용하여 확률 계산
    probs = F.softmax(logits, dim=1).squeeze().cpu().numpy()

    # 각 클래스에 대한 확률 반환
    return probs

def analyze_post_risk(post, model, tokenizer, label_map_inv):
    # 게시글 토큰화
    tokens = tokenizer.tokenize(post)

    # 각 토큰의 위험도 예측
    probs = classify_sentence(post, model, tokenizer)

    # 각 카테고리 및 예측 확률 분석
    category_probabilities = {}
    for label, prob in zip(label_map_inv.values(), probs):
        category_probabilities[label] = prob

    # 가장 높은 확률을 가진 카테고리와 그 확률을 찾음
    max_prob_category = max(category_probabilities, key=category_probabilities.get)
    max_prob = category_probabilities[max_prob_category]

    return max_prob_category, max_prob, category_probabilities

# 게시글 입력
# post = "재배강의 문의 및 합법화 면왭 활동에 대한 아이디어나 도움 주실분, 아동성범죄 영상도 팝니다"
# post = "페도필리아가 만든 사이트인데 사진 업로드 가능"
# post = "웹캠으로 디폴트 암호 안바꾸고 쭉 사용하는 집은 볼 수 있었음. 부부들 잠자리 확인 가능"

label_map_inv = {v: k for k, v in label_map.items()}

# 위험도 분석

for i in range(3):
    post = input("게시글 입력 : ")
    max_prob_category, max_prob, category_probabilities = analyze_post_risk(post, model, tokenizer, label_map_inv)
    if (max_prob_category == 'normal'):
        max_prob = 1 - max_prob
    # 결과 출력
    print("******RISK REPORT******")
    print(f"**위험 카테고리: {max_prob_category}\t**")
    print(f"**위험도 : {max_prob:.2f}%\t**")
    print("**각 카테고리별 확률:\t**")
    for category, prob in category_probabilities.items():
        print(f"**\t{category}: {prob:.2f}\t**")
    print("*************************")