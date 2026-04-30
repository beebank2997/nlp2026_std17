# 1.Sinusodal Position Encoding 
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
    d_k =q.shape[-1]
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


#3. multi-head attention (การมองหลายมุมมอง) ช่วยให้โมเดลเข้าใจความสัมพันธ์หลายรูปแบบพร้อมกัน
class MultiHeadAttentionSimple :
    def __init__(self, d_model=16 , n_heads =4):
        self.n_heads =n_heads
        self.d_k = int(d_model // n_heads)
    def split_heads(self, x):
        batch , seq_len , d_model = x.shape
        x = x.reshape(batch,seq_len,self.n_heads,self.d_k)
        return x.transpose(0,2,1,3)

    def combine_heads(self, x):
        batch,heads.seq_len,d_k = x.shape
        x = x.transpose(0,2,1,3)
        return x.reshape(batch,seq_len,heads*d_k)
    
    def forward(self, x):
        #1. linear projection ก่อนแยก heads
        Q = np.matmul(x, self.W_q)
        K = np.matmul(x, self.W_k)
        V = np.matmul(x, self.W_v)
        #2. แยกออกเป็น 4 หัว (1,4,5,4)
        Q = self.split_heads(Q)
        K = self.split_heads(K)
        V = self.split_heads(V)
        # attention แต่ละ head-reshape เพื่อให้ batch head รวมกัน
        bath = x.shape[0]
        Q_r = Q.reshape(batch * self.n_heaads, x.shape[1], self.d_k) #batch = 1 --> (4,5,4)
        K_r = K.reshape(batch * self.n_heaads, x.shape[1], self.d_k)
        V_r = V.reshape(batch * self.n_heaads, x.shape[1], self.d_k)
        attn_out,attn_weight = scale_dot_product_attention(Q_r,K_r,V_r)
        #reshape attention weight --> (batch , head , seq ,seq)
        attn_weights = attn_weights.reshape(batch,self.n_heads, x.shape[1], x.shape[1])
        #step 4: รวม heads กลับ
        attn_out = attn_out.reshape(batch,self.n_heads, x.shape[1], self.d_k)
        concat = self.combine_heads(attn_out)
        #step5 : output projection
        output = np.matmal(concat,self.W_o)
        return output , attn_weight
#จำลอง input ขนาด 16 - มิติ (d) แบงเป็น 4 heads (Head ละ 4 มิติ)
input_data = np.random.randn(1,5,16)
mha = MultiHeadAttentionSimple()
heads = mha.split_heads(input_data)
print (f"Origin shape : {input_data.shape}")
print(f"Heads shape : {heads.shape} (Batch , Heads , Seq_len , Depth)")


#3.2 Postional Encoding Multi-Head Attention
d_model = 16
n_heads = 4
seq_len = 5
batch = 1
# Step1 : random input-vector Embeding
np.random.seed(17)
token_embeddings = np.random.randn(batch,seq_len_model1)
#Step2 : บวกค่าของ Positional Encoding
pe_encoder = SinusodalPositionEncoding(max_seq_len=10,d_model=d_model)
pe_encoder.show(seq_len)
x = token_embeddings + pe_encoder.ppe[:seq_len]

#step3 : multi head attention
mha = MultiHeadAttentionSimple(d_model=d_model, n_heads=n_heads)
out, attn_weights = mha.forward(x)
#แสดงผล
print(f"--Multihead Attension ")
print(f"inout shpe : {x.shape}")
print(f"output shape : {output.shape}")
print(f"weight shape : {attn_weights.shape} -> (batch,heads. seq_q , seq_k)")
for h in range (n_heads):
    print(f"\nHead {h+1} attention weight:")
    for row in attn_weights[0,h]:
        print(" "," ".join(f"{v: .3f}" for v in row))

#4. Transformer Encoder (BERT) and Decode (GPT)
#4.1 Feed Forward Network (FFN)
class FeedForward :
    def __init__(self, d_model, d_ff=None , seed = 42):
        rng =  np.random.RandomState(seed)
        d_ff = d_ff or d_model * 4
        s = np.sqrt(2.0/d_model)
        