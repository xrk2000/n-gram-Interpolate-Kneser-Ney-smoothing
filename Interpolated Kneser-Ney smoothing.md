# Interpolated Kneser-Ney smoothing

## 1.语言模型

作业中使用 3-gram 语言模型，包含了 2-gram 和 1-gram 项。全部内容都个人自主独立实现。

## 2.平滑方式

1. 采用 Interpolated Kneser-Ney smoothing。
2. 参考文献
   - [1] Heafield K ,  Pouzyrevsky I ,  Clark J H , et al. Scalable Modified Kneser-Ney Language Model Estimation[J].  2013.
   - [2]D Jurafsky. Speech and Language Processing: An Introduction to Natural Language Processing, Computational Linguistics, and Speech Recognition[M]. 人民邮电出版社, 2010.
   - [3]马尚先生. 基于统计语言模型的平滑算法[EB/OL]. [2021-10-17]. https://zhuanlan.zhihu.com/p/111963567.

## 3.平滑理论

1.  **绝对减值法**

   ​		Add-one 和add-K 算法的策略本质是说将一些频繁出现的N-grams 的概率均匀出一些给那些没有出现的N-grams上，最终要保证所有可能性的概率之和等于1。那现在的问题是到底是均匀出多少来？Church & Gale (1991) 设计了一种非常巧妙的方法。首先他们在一个留存语料库（held-out corpus）考察那些在训练集中出现了4次的bigrams出现的次数。具体来说，他们首先在一个有2200万词的留存语料库中检索出所有出现了4次的bigrams（例如：“pick up”，“kick off”等），然后再从一个同样有2200万词的训练集中，分别统计这些bigrams出现的次数，例如“pick up=5”，“kick off=4”最后取个平均。同样的统计出现1到9次的bigrams 分别在training set 和held-out 中出现的次数。最终，平均下来，不同频次的词相差在0.7左右。$^{[3]}$

2. **修正计数量 (a) Adjusting Counts**

   $假设我们使用n-gram的最高阶数是N，用 w_m^n表示w_mw_{m+1}...w_{n}，c(w_m^n)表示w_m^n在样本中出现的$$次数，用修正后的计数量a来代替c。^{[1]}$
   $$
   a(w_m^n)=
   \left\{
   \begin{aligned}
   c(w_m^n) &  & n-m+1=N，或w_m为<s> \\
   |v : c(vw_m^n)>0| &  & 其它 \\
   \end{aligned}
   \right.
   $$
   由上可知修正计数量是通过按后缀顺序排序的$^{[1]}$，我在实现时对计数后的 2-gram 与 3-gram 进行了重排列（见 count.py）。

3. **绝对减值量 (d)** **Absolute Discounting**

   将 dev_set.txt 中的数据当成 heldout set 来使用，注意到 train : dev = 8:1，故将其计数量均乘上 8。实现思路见参考文献$^{[2]}$，取 1~10项（忽略0，因为 <unk>太多），实现细节见 dev_d.py，结果放在 immediate/held_out/dev.txt，如下所示（保留两位小数）：

   | Bigram count in training set | Bigram count in heldout set | delta |
   | ---------------------------- | --------------------------- | ----- |
   | 1                            | 0.30                        | 0.70  |
   | 2                            | 1.09                        | 0.91  |
   | 3                            | 1.96                        | 1.04  |
   | 4                            | 2.90                        | 1.10  |
   | 5                            | 3.82                        | 1.18  |
   | 6                            | 4.73                        | 1.27  |
   | 7                            | 5.76                        | 1.24  |
   | 8                            | 6.50                        | 1.50  |
   | 9                            | 7.50                        | 1.50  |
   | 10                           | 8.52                        | 1.48  |

   舍去第一行，取平均得 d = 1.29（保留两位小数）。

4. **归一化 Normalization**

   归一化计算伪概率 u 以及 回退值  b$^{[1][2]}$。
   $$
   u(w_n|w_m^{n-1}) = \frac{max(a(w_m^n)-d,0)}{\sum_xa(w_m^{n-1}x)} \\
   b(w_m^{n-1}) = \frac{d*|\{x:a(w_m^{n-1}x)>0\}|}{\sum_xa(w_m^{n-1}x)}
   $$
   其中 u, b 的实现细节见 u.py 与 b.py，计算结果放在 model/u 与 model/b 中。

5. **根据递归函数进行插值计算概率 (p)  Interpolation according to the recursive equation**
   $$
   p(w_n|w_m^{n-1}) = u(w_n|w_m^{n-1}) + b(w_m^{n-1})p(w_n|w_{m+1}^{n-1})
   $$
   递归终止于 1-gram:
   $$
   p(w_n) = u(w_n) + b(ε)/|V_1|
   $$
   其中 $|V_1|$ 是词汇 vocabulary 即 1-gram 的种类数，ε 是空串。
   $$
   \begin{eqnarray*}
   && b(ε) = \frac{d*|\{x:a(x)>0\}|}{\sum_xa(x)}  \\
   && \ \ \ \ \ \ \  = d \frac{|\{x:|v:c(vx)>0|>0\}|} 
   {\sum_x|v : c(vx)>0|} \\
   && \ \ \ \ \ \ \  = d \frac{|\{x:(vx) \ is \ in \ training \ set\}|}{|V_2|}
   \end{eqnarray*}
   $$

## 4.代码实现

- 已上传至 https://github.com/xrk2000/n-gram-Interpolate-Kneser-Ney-smoothing/tree/master

- 纯用 python 实现

- immediate 文件夹下储存了计算过程中的中间结果文件，用 txt格式存储，共 540MB，若进一步优化可用二进制文件节省时间。

- 语言模型在 model 文件夹下，储存了 u1.txt, u2.txt, u3.txt, b1.txt, b2.txt,  p1.txt, p2.txt，共 477 MB。

- 测试 PPL 用到中间文件，只需运行 testppl.py 即在终端输出 PPL。

- 依次运行 count.py, dev_d.py,  u.py,  b.py, p.py 即可生成所有中间结果文件。

  ![](C:\Users\31475\Desktop\思维导图.png)

## 5.PPL结果

```python
ppl = 178.25
```

结果分析：数据集比较大，这个 PPL 和其他同学相比好像偏小？

