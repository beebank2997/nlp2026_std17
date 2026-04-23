#การตัดคำ Tokenization + custom_Dict
import re
LEGAL_KEYWORD = ["ลำเมิดสิทธิบัตร","เครื่องหมายการค้า","ลิขสิทธิ๋","การกระทำความผิด"]
def legal_tokenizer(text):
    #ใช้ regex จัดการเบื้องต้นแล้วค่อยตัดส่วนที่เหลือ
    compound = '|'.join(
        map(re.escape,sorted(LEGAL_KEYWORD,key = len, reverse=True)))
    pattern = compound + r"|u0E00-\u0E7F"+ r"|[a-zA-Z0-9]"
    tokens = re.findall(pattern, text)
    return tokens




test_text = "จำเลยรักกระทำความผิดฐานละเมิดสิทธิบัตรและเครื่องหมายการค้า"
tokens = legal_tokenizer(test_text)
print(f"input: {test_text}")
print(f"output: {tokens}")

#ตัดด้วย Deep learning (deepcut)
# import deepcut
# print(f"output: {deepcut.tokenize(test_text)}")