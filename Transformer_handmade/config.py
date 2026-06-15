
# ====== Transformer configuration parameters ======

N = 6           # number of layers
h = 8           # number of heads
d_model = 512   # dimension of the model
d_key = 64      # dimension of the key vectors
d_value = 64    # dimension of the value vectors
d_ff = 2048     # dimension of the feedforward network
dropout = 0.1           # dropout rate
steps = 10000           # number of training steps
batch_size = 64         # batch size
learning_rate = 1e-4    # learning rate
beta1 = 0.9             # beta1 for Adam optimizer
beta2 = 0.98            # beta2 for Adam optimizer
epsilon = 1e-9          # epsilon for Adam optimizer
max_seq_len = 512       # maximum sequence length
epsilon1s = 0.1         # epsilon for label smoothing