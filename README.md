# Word2Vec from Scratch Using NumPy

### Description
This repository contains a pure NumPy Python implementation of the Word2Vec Machine Learning model. 

### Implementation
- **Architecture:** Skip-gram (predicting context words based on the center word).

- **Optimization:** Negative sampling + 3/4 power noise distribution heuristic, used to boost the chance of rarer words appearing as negative samples.
- **Vectorization:** Heavy use of NumPy array operations to eliminate slow Python `for` loops.
- 
### Challenges & Optimizations
- **Numerical Stability (The `log(0)` Problem):** During backpropagation, the sigmoid function outputs converged to exactly `0.0` or `1.0` due to floating-point precision limits. When calculating the cross-entropy loss ($E$), it calculated `np.log(0)`, which in turn threw a `divide by zero` warning and resulted in an `Inf` total loss. 
  * **Fix:** I applied `np.clip()` to the sigmoid outputs (bounding them between `1e-15` and `1 - 1e-15`) right before the log calculation, ensuring numerical stability without impacting the gradient trajectory.

- **Performance Bottleneck (Vectorizing Negative Samples):** The naive approach to negative sampling used a Python `for` loop to draw $K$ noise words individually (`np.random.choice(V, p=noise_Distribution)`). Because the vocabulary $V$ is large, forcing NumPy to compute the cumulative sum of probabilities one by one created a massive bottleneck that heavily affected execution speed.
  * **Fix:** I vectorized the sampling by creating an array of size $K$ in a single operation (`size=K`) at the beginning of each context word iteration. Bypassing the nested Python loop and letting NumPy handle the batch sampling drastically reduced the epoch training time.

### Dataset
The model is trained on the **text8 dataset**, which is a standard Wikipedia corpus used by Mikolov et al. in the original Word2Vec paper. The model has over 1 million words, I used only 50 thousand for execution time and testing purposes. 

### How to Run
1. Clone the repository:
    git clone [https://github.com/TEODORCRISTESCU/Word2Vec-Numpy-Only.git](https://github.com/TEODORCRISTESCU/Word2Vec-Numpy-Only.git)

2. Install dependencies:
   pip install numpy kagglehub


### My results:

- Total loss for Epoch 1 is 1395278.70
- Total loss for Epoch 2 is 1242853.92
- Total loss for Epoch 3 is 1194931.81
- Total loss for Epoch 4 is 1161611.48
- Total loss for Epoch 5 is 1130536.63

### References
- Illustrated Word2Vec — https://jalammar.github.io/illustrated-word2vec/
- CS224N Lecture 1 — https://www.youtube.com/watch?v=rmVRLeJRkl4
- CS224N Lecture 2 — https://www.youtube.com/watch?v=gqaHkPEZAew
- word2vec Parameter Learning Explained (Rong, 2016) — https://arxiv.org/abs/1411.2738
- Mikolov et al. negative sampling paper — https://arxiv.org/abs/1310.4546
  

