
import numpy as np

class SinusodalPositionEncoding:
    def __init__(self, max_seq_len = 10, d_model = 16):
        pe = np.zeros((max_seq_len,d_model))
        pos = np.arange(max_seq_len).reshape(-1, 1)

        div = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))

        pe[:,0::2] = np.sin(pos * div)
        pe[:,1::2] = np.cos(pos * div)

        self.pe = pe

    def show(self,seq_len=5):
        print(f"Possition Encoding (First {seq_len} tokens)")
        for p in range(seq_len):
            print(f"Pos : {p} " + " ".join(f"{v:.2f}" for v in self.pe[p, :4]) + "...")

    def show_with_words(self,words):
        print(f"{'index':<7} | {'word':<10} | {'Positional Encoding (first 4 dims)':<40}")
        for i, word in enumerate(words):
            if i >= len(self.pe):
                break
            vec = self.pe[i,:4]
            vec_str = " ".join(f"{v:+.3f}" for v in vec)
            print(f"Pos_{i:<3} | {word:<10} | [{vec_str} ...]")

# การใช้งาน
words_list = ["You", "are", "a", "stupid", "human"]
pe = SinusodalPositionEncoding(max_seq_len=10, d_model=16)
pe.show_with_words(words_list)

    
    # pe = SinusodalPositionEncoding()
    # pe.show()

#2. Scale Dot-Product and padding Mask ให้ความยาวเอกสารเท่ากัน
def scale_dot_product_attention(Q,K,V, mask=None):
    d_k =Q.shape[-1]
    scores = np.matmul(Q, K.transpose(0,2,1)) / np.sqrt(d_k)
    if mask is not None:
        #กำหนดตำแหน่งที่เป็น 0 ใน mask มีค่า score=-inf
        scores = np.where(mask==0, -1e9, scores)
    weights = np.exp(scores-np.max(scores))
    weights /= weights.sum(axis=-1, keepdims=True)
    return weights @ V, weights
#จำลองข้อมูล 1 ประโยค 3 tokens,vector 4 มิติ
q= k = v = np.random.randn(1,3,4)
output , weights = scale_dot_product_attention(q,k,v)
print(f"---Attention Weights 3x3 Matrixs ----")
print(weights[0].round(2))
