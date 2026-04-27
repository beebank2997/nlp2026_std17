# การตัดคำ Tokenization + custom_Dict
import re
from pythainlp.tokenize import word_tokenize

LEGAL_KEYWORDS = ["ละเมิดสิทธิบัตร","เครื่องหมายการค้า","ลิขสิทธิ์","การกระทำความผิด"]

def legal_tokenizer(text):
    sorted_kw = sorted(LEGAL_KEYWORDS, key=len, reverse=True)
    placeholders = {}
    protected = text
    for i, kw in enumerate(sorted_kw):
        ph = f"__KW{i}__"
        if kw in protected:
            placeholders[ph] = kw
            protected = protected.replace(kw, ph)

    tokens_raw = word_tokenize(protected, engine="newmm", keep_whitespace=False)
    return [placeholders.get(t, t) for t in tokens_raw]


test_text = "จำเลยกระทำความผิดฐานละเมิดสิทธิบัตรและเครื่องหมายการค้า"
tokens = legal_tokenizer(test_text)


# การวัดค่าความกำกวม (Ambiguity Rate)
def calculate_baseline_ambiguity(text):
    matches = []
    for word in LEGAL_KEYWORDS:
        for m in re.finditer(re.escape(word), text):
            if m:
                matches.append((m.start(), m.end(), word))

    overlaps = 0
    for i in range(len(matches)):
        for j in range(i+1, len(matches)):
            if matches[i][0] < matches[j][1] and matches[j][0] < matches[i][1]:
                overlaps += 1

    return overlaps/len(matches) if matches else 0


# รัน baseline
sample_text = "คดีการละเมิดสิทธิบัตรและเครื่องหมายการค้า"
baseline_tokens = legal_tokenizer(sample_text)
baseline_rate = calculate_baseline_ambiguity(sample_text)

print("W1 Baseline Result:")
print(f"Token: {baseline_tokens}")
print(f"Baseline Ambiguity rate: {baseline_rate}")


# WangchanBERTa Pretrain
from transformers import AutoTokenizer

model_name = "airesearch/wangchanberta-base-att-spm-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def berta_tokenizer(text):
    tokens = tokenizer.tokenize(text)
    return [t.replace(" ", "") for t in tokens if t.replace(" ", "")]


def analyze_refined_ambiguity(text, legal_keywords):
    tokens = berta_tokenizer(text)
    frag_score = []
    for kw in legal_keywords:
        if kw in text:
            kw_tokens = berta_tokenizer(kw)
            fragment_ratio = len(kw_tokens)
            frag_score.append(fragment_ratio)

    avg_frag = (sum(frag_score)/len(frag_score) - 1) if frag_score else 0
    return min(avg_frag, 1.0)


# รัน refined
refined_tokens = berta_tokenizer(sample_text)
refined_rate = analyze_refined_ambiguity(sample_text, LEGAL_KEYWORDS)

print("--W1 : Refined With WangchanBERTa ---")
print(f"Tokens : {refined_tokens}")
print(f"New Ambiguity Fragmentation Rate : {refined_rate:.3f}")


# 2. Context-Aware Entity Extraction
def extract_legal_entities(text):
    entities = []
    if "สิทธิบัตร" in text:
        entities.append({"type":"IP_TYPE", "value":"PATENT","conf":0.95})
    if "ละเมิด" in text:
        entities.append({"type":"ACTION", "value":"INFRINGEMENT","conf":0.85})
    return entities


sample = "มีการละเมิดสิทธิบัตรเกิดขึ้นในเขตพื้นที่"
found = extract_legal_entities(sample)

for e in found:
    print(f"{e['type']} {e['value']} (confidence: {e['conf']})")


# 3. Feature Engineering (TF-IDF)
from sklearn.feature_extraction.text import TfidfVectorizer

corpus = [
    "ละเมิดสิทธิบัตร เครื่องหมายการค้า",
    "การกระทำความผิด ลิขสิทธิ์",
    "จำเลย ละเมิด ลิขสิทธิ์"
]

vectorizer = TfidfVectorizer(tokenizer=legal_tokenizer, token_pattern=None)
tfidf_matrix = vectorizer.fit_transform(corpus)


# 4. Physics Gate Weight
def compute_physics_gate_weight(entities):
    base_weight = 5.0
    for e in entities:
        if e['value'] == "PATENT":
            base_weight += 2.0
        if e['value'] == "INFRINGEMENT":
            base_weight += 1.5
    return min(base_weight, 10.0)


weight = compute_physics_gate_weight(found)

print("--Physics Gate Bridge --")
print(f"Legal Context Weight : {weight:.2f}/10")
print(f"Status: {'High Alert - Trigger Sensor' if weight >= 7 else 'Normal Monitoring'}")