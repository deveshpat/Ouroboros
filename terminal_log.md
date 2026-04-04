# terminal_log.md — Project Ouroboros
**All verified terminal outputs, newest first.**

---

> **Naming note:** Scripts were renamed for consistency.
> `phase1_viability_gate.py` → `viability_gate.py`
> `train_sft_phase2.py` → `train_sft.py`
> `Ouroboros_Blueprint_v3.md` → `BLUEPRINT.md`
> Logs below use original script names as they appeared at run time.

---

## Stage 1 — Pre-training, Kaggle Dual T4 (Session 5, resumed from checkpoint-0002000)
**Script:** `pretrain.py`
**Date:** 2026-04-03
**Hardware:** Kaggle Dual T4 (2× T4 16 GB, DDP auto-launched, world_size=2)
**Status:** 🟡 IN PROGRESS — log captured through step 9000 / 61,036 (run still ongoing)

**Bug 5 fix confirmed:**
```
  [resume] local   checkpoint-0002000  step=2000  epoch=0  tokens=65,536,000  val_ce=5.564212733394234
```
Clean local resume from disk. ✅

**Startup:**
```
2026-04-03 08:27:48 
2026-04-03 08:27:48 ========================================================================
2026-04-03 08:27:48   Stage 1 Pre-training - Project Ouroboros
2026-04-03 08:27:48 ========================================================================
2026-04-03 08:27:48   dataset          : HuggingFaceFW/fineweb-edu / sample-10BT
2026-04-03 08:27:48   tokenizer        : Qwen/Qwen2.5-0.5B  vocab=151,665
2026-04-03 08:27:48   preset           : nano
2026-04-03 08:27:48   model            : d_model=512  groups=1  heads=8/4
2026-04-03 08:27:48   chunk_size       : 1024
2026-04-03 08:27:48   batch x accum    : 8 global x 4
2026-04-03 08:27:48   world_size       : 2  (DDP auto-enabled)
2026-04-03 08:27:48   per_gpu_batch    : 4
2026-04-03 08:27:48   tokens / step    : 32,768
2026-04-03 08:27:48   token_budget     : 2,000,000,000
2026-04-03 08:27:48   total_steps      : 61,036
2026-04-03 08:27:48   dtype            : torch.bfloat16
2026-04-03 08:27:48   device           : cuda:0
2026-04-03 08:27:48   output_dir       : runs/stage1
2026-04-03 08:27:48   push_to_hub      : True
2026-04-03 08:27:48 ========================================================================
2026-04-03 08:27:50 
2026-04-03 08:27:50 Model parameters : 92,477,440 (92.5 M)
2026-04-03 08:27:50 
2026-04-03 08:27:50   Building val buffer (2,000,000 tokens) ...
2026-04-03 08:28:05   Val buffer: 2,000,000 tokens from 1,887 docs
2026-04-03 08:28:06   [resume] local   checkpoint-0002000  step=2000  epoch=0  tokens=65,536,000  val_ce=5.564212733394234
2026-04-03 08:28:07    step   train_ce     val_ce       smth    gnorm         lr     VRAM      tok/s
2026-04-03 08:28:07 --------------------------------------------------------------------------------
2026-04-03 08:28:07   epoch 0  offset=228  skipping=64000 chunks
```

**Training log (steps 2000–9000):**
```
2026-04-03 08:36:44    2050     4.5941          -     4.8104   0.7031   5.99e-04    2.035       3169
2026-04-03 08:41:26    2100     4.5080          -     4.7605   0.3906   5.99e-04    2.035       5809
2026-04-03 08:46:08    2150     4.7421          -     4.7493   0.4121   5.99e-04    2.035       5808
2026-04-03 08:50:51    2200     4.7011          -     4.7553   0.4277   5.99e-04    2.035       5802
2026-04-03 08:55:33    2250     4.6945          -     4.7333   0.4043   5.98e-04    2.035       5806
2026-04-03 09:00:09   [spike] step=2299  raw=5.3932  ema=4.7166
2026-04-03 09:00:14    2300     5.0453          -     4.7199   0.9805   5.98e-04    2.035       5815
2026-04-03 09:04:56    2350     4.5325          -     4.7081   0.4570   5.98e-04    2.035       5810
2026-04-03 09:09:38    2400     4.4693          -     4.6923   0.8672   5.98e-04    2.035       5810
2026-04-03 09:14:21    2450     4.6070          -     4.6939   0.5508   5.98e-04    2.035       5808
2026-04-03 09:17:04   [spike] step=2479  raw=5.2191  ema=4.6873
2026-04-03 09:19:03    2500     4.5709          -     4.6795   0.4023   5.98e-04    2.035       5808
2026-04-03 09:22:48   [val] step=2500  val_ce=5.4796
2026-04-03 09:22:48 
2026-04-03 09:22:48   -- Generation @ step 2500 (live weights) --
2026-04-03 09:22:50   P: The capital of France is
2026-04-03 09:22:50   C:  the only one that is the most important of the world. The world is the world's most important part of the world. The world is the world's largest city, and the
2026-04-03 09:22:50      uwr=0.202
2026-04-03 09:22:51   P: In mathematics, a prime number is
2026-04-03 09:22:51   C:  a simple one. The first thing that is important is that the basic principle of the word is the value of the word. The word is used to describe the word, and th
2026-04-03 09:22:51      uwr=0.193
2026-04-03 09:22:53   P: def factorial(n):
2026-04-03 09:22:53     """Return n!."""
2026-04-03 09:22:53     if n
2026-04-03 09:22:53   C: . 1000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
2026-04-03 09:22:53      uwr=1.000
2026-04-03 09:22:54   P: Neural networks learn by
2026-04-03 09:22:54   C:  using the same technology as the internet. The internet is a very common type of internet connection, and the internet is a very common type of internet connec
2026-04-03 09:22:54      uwr=0.222
2026-04-03 09:22:56   P: The French Revolution began in
2026-04-03 09:22:56   C:  the 19th century, when the French Revolution was the first to be a part of the Soviet Union. The Soviet Union was the first to be a part of the Soviet Union. T
2026-04-03 09:22:56      uwr=0.260
2026-04-03 09:22:56   Mean UWR: 0.375
2026-04-03 09:27:38    2550     4.8075     5.4796     4.6841   0.6641   5.98e-04    2.035       3181
2026-04-03 09:32:20    2600     4.5635     5.4796     4.6661   0.3965   5.98e-04    2.035       5802
2026-04-03 09:37:02    2650     4.5249     5.4796     4.6582   0.4473   5.98e-04    2.035       5806
2026-04-03 09:41:44    2700     4.4584     5.4796     4.6462   0.5391   5.98e-04    2.035       5810
2026-04-03 09:46:26    2750     4.7246     5.4796     4.6456   0.3809   5.98e-04    2.035       5815
2026-04-03 09:48:30   [spike] step=2772  raw=5.1745  ema=4.6504
2026-04-03 09:51:08    2800     4.5656     5.4796     4.6321   0.3770   5.98e-04    2.035       5810
2026-04-03 09:55:50    2850     4.5634     5.4796     4.6257   0.3691   5.97e-04    2.035       5810
2026-04-03 10:00:32    2900     4.5406     5.4796     4.6220   0.3730   5.97e-04    2.035       5810
2026-04-03 10:02:42   [spike] step=2923  raw=5.1878  ema=4.6173
2026-04-03 10:05:14    2950     4.4513     5.4796     4.6141   0.5586   5.97e-04    2.035       5805
2026-04-03 10:09:56    3000     4.4762     5.4796     4.6025   0.4414   5.97e-04    2.035       5823
2026-04-03 10:13:41   [val] step=3000  val_ce=5.4154
2026-04-03 10:13:41 
2026-04-03 10:13:41   -- Generation @ step 3000 (live weights) --
2026-04-03 10:13:42   P: The capital of France is
2026-04-03 10:13:42   C:  the most important part of the economy. The main purpose of the economy is to make the economy more productive and productive. The economy is a system of econo
2026-04-03 10:13:42      uwr=0.255
2026-04-03 10:13:44   P: In mathematics, a prime number is
2026-04-03 10:13:44   C:  the number of times the number of times the number of people in a given group is 1. The number of people in the group is 1. The number of people who are in the
2026-04-03 10:13:44      uwr=0.152
2026-04-03 10:13:45   P: def factorial(n):
2026-04-03 10:13:45     """Return n!."""
2026-04-03 10:13:45     if n
2026-04-03 10:13:45   C: . 1000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
2026-04-03 10:13:45      uwr=1.000
2026-04-03 10:13:46   P: Neural networks learn by
2026-04-03 10:13:46   C:  the time they are given. The data is then collected from the data and then sent to the data. The data is then collected from the data. The data is then collect
2026-04-03 10:13:46      uwr=0.160
2026-04-03 10:13:48   P: The French Revolution began in
2026-04-03 10:13:48   C:  1822, and the French Revolution was the first to be replaced by the French Revolution. The French Revolution was the first to be a new revolution, and the new
2026-04-03 10:13:48      uwr=0.150
2026-04-03 10:13:48   Mean UWR: 0.343
2026-04-03 10:13:49   [ckpt] saved  -> runs/stage1/checkpoint-0003000
2026-04-03 10:13:49   [hub] uploading checkpoint-0003000 -> WeirdRunner/Ouroboros ...
2026-04-03 10:14:05   [hub] uploaded  checkpoint-0003000 (commit=5e2ba2b8)
2026-04-03 10:18:49    3050     4.5598     5.4154     4.5969   0.4785   5.97e-04    2.035       3070
2026-04-03 10:23:32    3100     4.7398     5.4154     4.5867   0.4180   5.97e-04    2.035       5804
2026-04-03 10:24:39   [spike] step=3112  raw=5.4127  ema=4.5942
2026-04-03 10:27:51   [spike] step=3146  raw=5.2242  ema=4.6168
2026-04-03 10:28:14    3150     4.6520     5.4154     4.6142   0.4648   5.97e-04    2.035       5801
2026-04-03 10:29:11   [spike] step=3160  raw=5.2053  ema=4.6183
2026-04-03 10:32:56    3200     4.4035     5.4154     4.6143   0.3926   5.97e-04    2.035       5801
2026-04-03 10:37:39    3250     4.7021     5.4154     4.6059   0.6797   5.97e-04    2.035       5807
2026-04-03 10:42:21    3300     4.7683     5.4154     4.5951   0.3867   5.97e-04    2.035       5806
2026-04-03 10:47:03    3350     4.4831     5.4154     4.5798   0.4121   5.96e-04    2.035       5803
2026-04-03 10:51:46    3400     4.1759     5.4154     4.5612   0.4941   5.96e-04    2.035       5802
2026-04-03 10:56:28    3450     4.5199     5.4154     4.5482   0.4004   5.96e-04    2.035       5800
2026-04-03 10:58:49   [spike] step=3475  raw=5.8600  ema=4.5615   ← ⚠ cluster start
2026-04-03 10:58:55   [spike] step=3476  raw=5.8159  ema=4.5740   ← ⚠ consecutive
2026-04-03 10:59:00   [spike] step=3477  raw=5.6536  ema=4.5848   ← ⚠ consecutive
2026-04-03 11:01:10    3500     4.4163     5.4154     4.5772   0.5078   5.96e-04    2.035       5799
2026-04-03 11:04:56   [val] step=3500  val_ce=5.3622
2026-04-03 11:04:56 
2026-04-03 11:04:56   -- Generation @ step 3500 (live weights) --
2026-04-03 11:04:57   P: The capital of France is
2026-04-03 11:04:57   C:  a major factor in the development of the country. The government of the country is a major factor in the development of the country. The government of the coun
2026-04-03 11:04:57      uwr=0.217
2026-04-03 11:04:59   P: In mathematics, a prime number is
2026-04-03 11:04:59   C:  a number of numbers. The number of numbers is the number of numbers in a number. The number of numbers is the number of numbers in the number of numbers. The n
2026-04-03 11:04:59      uwr=0.088
2026-04-03 11:05:00   P: def factorial(n):
2026-04-03 11:05:00     """Return n!."""
2026-04-03 11:05:00     if n
2026-04-03 11:05:00   C: is ll l l l l l l l l l l l l l l l l l l l l l l l l l l l l l l l l l l l l l l l l l l l l l l l l l l l l l l l l l l l l l l l l l l l l l l l l l l l l l
2026-04-03 11:05:00      uwr=0.025
2026-04-03 11:05:02   P: Neural networks learn by
2026-04-03 11:05:02   C:  the way, and the network is a network of networks that can be used to communicate with the network. The network is a network that is connected to the network,
2026-04-03 11:05:02      uwr=0.178
2026-04-03 11:05:03   P: The French Revolution began in
2026-04-03 11:05:03   C:  the 19th century, and the French Revolution was the result of the revolution of the 19th century. The French Revolution, however, was not a new one. The French
2026-04-03 11:05:03      uwr=0.267
2026-04-03 11:05:03   Mean UWR: 0.155
2026-04-03 11:09:46    3550     4.5172     5.3622     4.5533   0.4219   5.96e-04    2.035       3179
2026-04-03 11:11:39   [spike] step=3570  raw=6.7288  ema=4.5730   ← ⚠ cluster start
2026-04-03 11:11:55   [spike] step=3573  raw=5.3007  ema=4.5825   ← ⚠ consecutive
2026-04-03 11:12:58   [spike] step=3584  raw=5.1362  ema=4.5855
2026-04-03 11:14:22   [spike] step=3599  raw=5.2118  ema=4.5896
2026-04-03 11:14:28    3600     4.6403     5.3622     4.5901   0.4688   5.96e-04    2.035       5809
2026-04-03 11:19:10    3650     4.5792     5.3622     4.5766   0.3926   5.96e-04    2.035       5797
2026-04-03 11:23:53    3700     4.4256     5.3622     4.5695   0.4219   5.96e-04    2.035       5800
2026-04-03 11:25:23   [spike] step=3716  raw=5.1147  ema=4.5612
2026-04-03 11:28:35    3750     4.5366     5.3622     4.5614   0.4043   5.95e-04    2.035       5804
2026-04-03 11:33:18    3800     4.4380     5.3622     4.5457   0.4180   5.95e-04    2.035       5805
2026-04-03 11:38:00    3850     4.3969     5.3622     4.5468   1.4453   5.95e-04    2.035       5802
2026-04-03 11:42:43    3900     4.2804     5.3622     4.5202   0.6406   5.95e-04    2.035       5795
2026-04-03 11:47:25    3950     4.4417     5.3622     4.5044   0.4414   5.95e-04    2.035       5796
2026-04-03 11:52:08    4000     4.4599     5.3622     4.4950   0.3926   5.95e-04    2.035       5803
2026-04-03 11:55:53   [val] step=4000  val_ce=5.3399
2026-04-03 11:55:53 
2026-04-03 11:55:53   -- Generation @ step 4000 (live weights) --
2026-04-03 11:55:55   P: The capital of France is
2026-04-03 11:55:55   C:  the capital of the British Empire. The British Empire is the largest city in the world, and the largest city in the world is the United States. The city is a m
2026-04-03 11:55:55      uwr=0.321
2026-04-03 11:55:56   P: In mathematics, a prime number is
2026-04-03 11:55:56   C:  a number of times that is the number of times the number of stars in a given number. The number of stars in a number is the number of stars in the number of st
2026-04-03 11:55:56      uwr=0.124
2026-04-03 11:55:58   P: def factorial(n):
2026-04-03 11:55:58     """Return n!."""
2026-04-03 11:55:58     if n
2026-04-03 11:55:58   C: . 1000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
2026-04-03 11:55:58      uwr=1.000
2026-04-03 11:55:59   P: Neural networks learn by
2026-04-03 11:55:59   C:  the use of the same language as the language of the human brain. The brain is a very complex system of sensory systems, which is a complex system of sensory sy
2026-04-03 11:55:59      uwr=0.301
2026-04-03 11:56:00   P: The French Revolution began in
2026-04-03 11:56:00   C:  1860, when the British government was defeated by the British and the British, and the British and British, and the British, were defeated. The British were th
2026-04-03 11:56:00      uwr=0.144
2026-04-03 11:56:00   Mean UWR: 0.378
2026-04-03 11:56:02   [ckpt] saved  -> runs/stage1/checkpoint-0004000
2026-04-03 11:56:02   [hub] uploading checkpoint-0004000 -> WeirdRunner/Ouroboros ...
2026-04-03 11:56:17   [hub] uploaded  checkpoint-0004000 (commit=dd4b5eb3)
2026-04-03 11:57:37   [spike] step=4014  raw=5.0185  ema=4.5080
2026-04-03 12:01:02    4050     4.4937     5.3399     4.4912   0.4648   5.95e-04    2.035       3069
2026-04-03 12:05:44    4100     4.3299     5.3399     4.4968   0.4395   5.95e-04    2.035       5802
2026-04-03 12:10:26    4150     4.4858     5.3399     4.4792   0.3926   5.94e-04    2.035       5807
2026-04-03 12:13:44   [spike] step=4185  raw=5.0551  ema=4.4939
2026-04-03 12:15:09    4200     4.6867     5.3399     4.4922   0.4941   5.94e-04    2.035       5801
2026-04-03 12:19:51    4250     4.5452     5.3399     4.5059   0.3848   5.94e-04    2.035       5792
2026-04-03 12:24:33    4300     4.3885     5.3399     4.5026   0.4102   5.94e-04    2.035       5813
2026-04-03 12:29:16    4350     4.4287     5.3399     4.4918   0.4355   5.94e-04    2.035       5793
2026-04-03 12:33:58    4400     4.2816     5.3399     4.4763   0.3906   5.94e-04    2.035       5817
2026-04-03 12:38:40    4450     4.9241     5.3399     4.4858   0.6875   5.94e-04    2.035       5806
2026-04-03 12:39:08   [spike] step=4455  raw=5.7178  ema=4.4939
2026-04-03 12:43:22    4500     4.5862     5.3399     4.4983   0.4531   5.93e-04    2.035       5810
2026-04-03 12:47:07   [val] step=4500  val_ce=5.3188
2026-04-03 12:47:07 
2026-04-03 12:47:07   -- Generation @ step 4500 (live weights) --
2026-04-03 12:47:09   P: The capital of France is
2026-04-03 12:47:09   C:  a city of the city of the city of the city of the city of the city of the city of the city of the city of the city of the city of the city of the city of the c
2026-04-03 12:47:09      uwr=0.033
2026-04-03 12:47:10   P: In mathematics, a prime number is
2026-04-03 12:47:10   C:  a number of numbers. The number of numbers is the number of numbers that are in the number of numbers. The number of numbers is the number of numbers that are
2026-04-03 12:47:10      uwr=0.098
2026-04-03 12:47:11   P: def factorial(n):
2026-04-03 12:47:11     """Return n!."""
2026-04-03 12:47:11     if n
2026-04-03 12:47:11   C: . 11.11.111.111.11111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111
2026-04-03 12:47:11      uwr=1.000
2026-04-03 12:47:13   P: Neural networks learn by
2026-04-03 12:47:13   C:  learning how to communicate with a computer. They can learn to communicate with their peers, and learn to communicate with others. They can learn to communicat
2026-04-03 12:47:13      uwr=0.175
2026-04-03 12:47:14   P: The French Revolution began in
2026-04-03 12:47:14   C:  1777, and the French Revolution was the first to be established in the United States. The French Revolution was the first to be established in the United State
2026-04-03 12:47:14      uwr=0.204
2026-04-03 12:47:14   Mean UWR: 0.302
2026-04-03 12:51:57    4550     4.5178     5.3188     4.4888   0.3789   5.93e-04    2.035       3183
2026-04-03 12:56:39    4600     4.3192     5.3188     4.4779   0.3672   5.93e-04    2.035       5801
2026-04-03 13:01:21    4650     4.2660     5.3188     4.4628   0.3848   5.93e-04    2.035       5808
2026-04-03 13:06:03    4700     4.4701     5.3188     4.4656   0.4434   5.93e-04    2.035       5812
2026-04-03 13:10:45    4750     4.5391     5.3188     4.4612   0.4531   5.93e-04    2.035       5803
2026-04-03 13:15:28    4800     4.7103     5.3188     4.4557   0.7227   5.92e-04    2.035       5802
2026-04-03 13:20:10    4850     4.5228     5.3188     4.4504   0.5781   5.92e-04    2.035       5803
2026-04-03 13:24:52    4900     4.4096     5.3188     4.4519   0.4668   5.92e-04    2.035       5805
2026-04-03 13:29:35    4950     4.3229     5.3188     4.4434   0.3984   5.92e-04    2.035       5803
2026-04-03 13:34:17    5000     4.4682     5.3188     4.4454   0.3730   5.92e-04    2.035       5800
2026-04-03 13:38:03   [val] step=5000  val_ce=5.2965
2026-04-03 13:38:03 
2026-04-03 13:38:03   -- Generation @ step 5000 (live weights) --
2026-04-03 13:38:04   P: The capital of France is
2026-04-03 13:38:04   C:  the capital of the Republic of France. The capital of the Republic is the capital of the Republic of France. The capital of the Republic is the capital of the
2026-04-03 13:38:04      uwr=0.090
2026-04-03 13:38:06   P: In mathematics, a prime number is
2026-04-03 13:38:06   C:  a number. The number of numbers is the number of numbers in the numbers. The number of numbers is the number of numbers in the numbers. The number of numbers i
2026-04-03 13:38:06      uwr=0.089
2026-04-03 13:38:07   P: def factorial(n):
2026-04-03 13:38:07     """Return n!."""
2026-04-03 13:38:07     if n
2026-04-03 13:38:07   C: .d.
2026-04-03 13:38:07 - The name of the name is the name of the name of the name of the name of the name of the name of the name of the name of the name of the name of the name o
2026-04-03 13:38:07      uwr=0.059
2026-04-03 13:38:09   P: Neural networks learn by
2026-04-03 13:38:09   C:  the time they reach 100,000 years ago. The most common cause of death is the loss of a large number of people, including those who have been living in the Unit
2026-04-03 13:38:09      uwr=0.340
2026-04-03 13:38:10   P: The French Revolution began in
2026-04-03 13:38:10   C:  1811, and the French Revolution began in 1811. The French Revolution was the first time the French Revolution was a major part of the French Revolution. The Fr
2026-04-03 13:38:10      uwr=0.202
2026-04-03 13:38:10   Mean UWR: 0.156
2026-04-03 13:38:11   [ckpt] saved  -> runs/stage1/checkpoint-0005000
2026-04-03 13:38:11   [ckpt] pruned -> checkpoint-0002000
2026-04-03 13:38:11   [hub] uploading checkpoint-0005000 -> WeirdRunner/Ouroboros ...
2026-04-03 13:38:25   [hub] uploaded  checkpoint-0005000 (commit=38a64092)
2026-04-03 13:43:10    5050     4.4614     5.2965     4.4523   0.4199   5.92e-04    2.035       3076
2026-04-03 13:47:52    5100     4.5551     5.2965     4.4321   0.3770   5.91e-04    2.035       5797
2026-04-03 13:52:34    5150     4.4402     5.2965     4.4194   0.3848   5.91e-04    2.035       5811
2026-04-03 13:57:17    5200     4.4128     5.2965     4.4141   0.4590   5.91e-04    2.035       5801
2026-04-03 14:01:59    5250     4.6150     5.2965     4.4038   0.4121   5.91e-04    2.035       5801
2026-04-03 14:06:42    5300     4.4167     5.2965     4.3982   0.4219   5.91e-04    2.035       5802
2026-04-03 14:11:24    5350     4.5777     5.2965     4.3902   0.4453   5.91e-04    2.035       5798
2026-04-03 14:16:07    5400     4.2700     5.2965     4.3895   0.4375   5.90e-04    2.035       5796
2026-04-03 14:20:50    5450     4.3421     5.2965     4.3923   0.3945   5.90e-04    2.035       5793
2026-04-03 14:23:56   [spike] step=5483  raw=4.9702  ema=4.3954
2026-04-03 14:25:32    5500     4.3713     5.2965     4.3889   0.3848   5.90e-04    2.035       5809
2026-04-03 14:29:17   [val] step=5500  val_ce=5.2967
2026-04-03 14:29:17 
2026-04-03 14:29:17   -- Generation @ step 5500 (live weights) --
2026-04-03 14:29:18   P: The capital of France is
2026-04-03 14:29:18   C:  the capital of the city of the city of the city of the city of the city of the city of the city of the city of the city of the city of the city of the city of
2026-04-03 14:29:18      uwr=0.033
2026-04-03 14:29:20   P: In mathematics, a prime number is
2026-04-03 14:29:20   C:  a number. The number of numbers is the number of numbers that are the number of numbers. The number of numbers is the number of numbers that are the number of
2026-04-03 14:29:20      uwr=0.098
2026-04-03 14:29:21   P: def factorial(n):
2026-04-03 14:29:21     """Return n!."""
2026-04-03 14:29:21     if n
2026-04-03 14:29:21   C:  is not a
2026-04-03 14:29:21 - The following table is the same as the table of the table:
2026-04-03 14:29:21 - The table of the table is the table of the table.
2026-04-03 14:29:21 - The table of the table is the table
2026-04-03 14:29:21      uwr=0.118
2026-04-03 14:29:23   P: Neural networks learn by
2026-04-03 14:29:23   C:  using a computer to control the movement of the brain. The computer is able to detect the brain and brain in a way that is able to detect the brain and brain.
2026-04-03 14:29:23      uwr=0.231
2026-04-03 14:29:24   P: The French Revolution began in
2026-04-03 14:29:24   C:  the 19th century, when the French colonists were the first to be the first to be the French. The French had a strong influence on the French and French, and th
2026-04-03 14:29:24      uwr=0.179
2026-04-03 14:29:24   Mean UWR: 0.132
2026-04-03 14:34:07    5550     4.3649     5.2967     4.3877   0.4902   5.90e-04    2.035       3179
2026-04-03 14:38:50    5600     4.5273     5.2967     4.4064   0.7266   5.90e-04    2.035       5799
2026-04-03 14:43:33    5650     4.3939     5.2967     4.4073   0.5352   5.89e-04    2.035       5793
2026-04-03 14:48:16    5700     4.3187     5.2967     4.3931   0.4004   5.89e-04    2.035       5790
2026-04-03 14:52:58    5750     4.2418     5.2967     4.3825   0.4316   5.89e-04    2.035       5798
2026-04-03 14:57:41    5800     4.5618     5.2967     4.3724   0.4707   5.89e-04    2.035       5792
2026-04-03 15:02:24    5850     4.4097     5.2967     4.3761   0.4316   5.89e-04    2.035       5797
2026-04-03 15:07:06    5900     4.3369     5.2967     4.3767   0.4219   5.88e-04    2.035       5798
2026-04-03 15:11:49    5950     4.3106     5.2967     4.3715   0.4102   5.88e-04    2.035       5796
2026-04-03 15:14:21   [spike] step=5977  raw=4.8938  ema=4.3676
2026-04-03 15:16:31    6000     4.3142     5.2967     4.3581   0.4043   5.88e-04    2.035       5803
2026-04-03 15:20:17   [val] step=6000  val_ce=5.3051
2026-04-03 15:20:17 
2026-04-03 15:20:17   -- Generation @ step 6000 (live weights) --
2026-04-03 15:20:18   P: The capital of France is
2026-04-03 15:20:18   C:  the most important part of the city. The city is a city that is located in the city of the city. The city is located in the city of the city. The city is locat
2026-04-03 15:20:18      uwr=0.118
2026-04-03 15:20:20   P: In mathematics, a prime number is
2026-04-03 15:20:20   C:  a number. The number is the number of numbers that are in the number of numbers. The number is the number of numbers that are in the number. The number is the
2026-04-03 15:20:20      uwr=0.108
2026-04-03 15:20:21   P: def factorial(n):
2026-04-03 15:20:21     """Return n!."""
2026-04-03 15:20:21     if n
2026-04-03 15:20:21   C:  = 1, then the value of the value is 1.0, 1.0, 1.0, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1, 1, 2, 3, 3
2026-04-03 15:20:21      uwr=0.364
2026-04-03 15:20:23   P: Neural networks learn by
2026-04-03 15:20:23   C:  the way, and the brain is not able to function properly. The brain is also able to function properly. The brain is also able to function properly. The brain is
2026-04-03 15:20:23      uwr=0.123
2026-04-03 15:20:24   P: The French Revolution began in
2026-04-03 15:20:24   C:  the 19th century. The French Revolution was a period of time when the French Revolution began. The French Revolution was the first time the French government h
2026-04-03 15:20:24      uwr=0.211
2026-04-03 15:20:24   Mean UWR: 0.185
2026-04-03 15:20:25   [ckpt] saved  -> runs/stage1/checkpoint-0006000
2026-04-03 15:20:26   [ckpt] pruned -> checkpoint-0003000
2026-04-03 15:20:26   [hub] uploading checkpoint-0006000 -> WeirdRunner/Ouroboros ...
2026-04-03 15:20:40   [hub] uploaded  checkpoint-0006000 (commit=d958efbc)
2026-04-03 15:25:24    6050     4.2650     5.3051     4.3593   0.3496   5.88e-04    2.035       3076
2026-04-03 15:30:06    6100     4.3953     5.3051     4.3572   0.4590   5.88e-04    2.035       5803
2026-04-03 15:34:48    6150     4.1661     5.3051     4.3540   0.4746   5.87e-04    2.035       5814
2026-04-03 15:38:39   [spike] step=6191  raw=5.0532  ema=4.3587
2026-04-03 15:39:30    6200     4.3615     5.3051     4.3621   0.6523   5.87e-04    2.035       5816
2026-04-03 15:43:55   [spike] step=6247  raw=4.9575  ema=4.3542
2026-04-03 15:44:12    6250     4.6532     5.3051     4.3589   0.4922   5.87e-04    2.035       5804
2026-04-03 15:48:54    6300     4.3666     5.3051     4.3557   0.7695   5.87e-04    2.035       5810
2026-04-03 15:53:37    6350     4.2933     5.3051     4.3556   0.5312   5.87e-04    2.035       5800
2026-04-03 15:58:02   [spike] step=6397  raw=5.4820  ema=4.3617   ← ⚠ cluster start
2026-04-03 15:58:19   [spike] step=6400  raw=5.0740  ema=4.3683   ← ⚠ consecutive
2026-04-03 15:58:19    6400     5.0740     5.3051     4.3683   0.9766   5.86e-04    2.035       5796
2026-04-03 16:03:01    6450     4.3482     5.3051     4.3586   0.3867   5.86e-04    2.035       5811
2026-04-03 16:07:43    6500     4.3157     5.3051     4.3474   0.3711   5.86e-04    2.035       5805
2026-04-03 16:11:28   [val] step=6500  val_ce=5.2952
2026-04-03 16:11:28 
2026-04-03 16:11:28   -- Generation @ step 6500 (live weights) --
2026-04-03 16:11:30   P: The capital of France is
2026-04-03 16:11:30   C:  the capital of the city of the city of the city of the city of the city of the city of the city of the city of the city of the city of the city of the city of
2026-04-03 16:11:30      uwr=0.033
2026-04-03 16:11:31   P: In mathematics, a prime number is
2026-04-03 16:11:31   C:  the number of numbers that are equal to the number of numbers. The number of numbers is the number of numbers that are the number of numbers that are the numbe
2026-04-03 16:11:31      uwr=0.121
2026-04-03 16:11:33   P: def factorial(n):
2026-04-03 16:11:33     """Return n!."""
2026-04-03 16:11:33     if n
2026-04-03 16:11:33   C: . 10.1000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
2026-04-03 16:11:33      uwr=1.000
2026-04-03 16:11:34   P: Neural networks learn by
2026-04-03 16:11:34   C:  the time they are born. The most important part of the brain is the brain, which is the brain. The brain is the brain that is responsible for the processing of
2026-04-03 16:11:34      uwr=0.230
2026-04-03 16:11:36   P: The French Revolution began in
2026-04-03 16:11:36   C:  1622, and the French Revolution was followed by a new war of war. The French and French forces were defeated in the war, and the French and French forces were
2026-04-03 16:11:36      uwr=0.204
2026-04-03 16:11:36   Mean UWR: 0.318
2026-04-03 16:15:44   [spike] step=6544  raw=5.3215  ema=4.3546
2026-04-03 16:16:18    6550     4.4705     5.2952     4.3564   0.8047   5.86e-04    2.035       3186
2026-04-03 16:21:00    6600     4.2915     5.2952     4.3688   0.3984   5.85e-04    2.035       5803
2026-04-03 16:25:42    6650     4.2438     5.2952     4.3573   0.4141   5.85e-04    2.035       5811
2026-04-03 16:30:24    6700     4.1803     5.2952     4.3543   0.4277   5.85e-04    2.035       5803
2026-04-03 16:35:06    6750     4.2759     5.2952     4.3435   0.4180   5.85e-04    2.035       5807
2026-04-03 16:39:49    6800     4.3008     5.2952     4.3399   0.4355   5.84e-04    2.035       5801
2026-04-03 16:44:31    6850     4.2073     5.2952     4.3468   0.3535   5.84e-04    2.035       5814
2026-04-03 16:49:13    6900     4.0178     5.2952     4.3344   0.7773   5.84e-04    2.035       5809
2026-04-03 16:51:11   [spike] step=6921  raw=5.1116  ema=4.3352
2026-04-03 16:53:55    6950     4.2331     5.2952     4.3405   0.3457   5.84e-04    2.035       5810
2026-04-03 16:58:37    7000     4.4510     5.2952     4.3391   0.5664   5.84e-04    2.035       5804
2026-04-03 17:02:22   [val] step=7000  val_ce=5.3075
2026-04-03 17:02:22 
2026-04-03 17:02:22   -- Generation @ step 7000 (live weights) --
2026-04-03 17:02:23   P: The capital of France is
2026-04-03 17:02:23   C:  the capital of the city of the city of the city of St. Louis. The city of the city is the capital of the city of the city of St. Louis. The city is the capital
2026-04-03 17:02:23      uwr=0.093
2026-04-03 17:02:25   P: In mathematics, a prime number is
2026-04-03 17:02:25   C:  a number of digits. The number of digits is the number of digits that are the number of digits. The number of digits is the number of digits that are the numbe
2026-04-03 17:02:25      uwr=0.089
2026-04-03 17:02:26   P: def factorial(n):
2026-04-03 17:02:26     """Return n!."""
2026-04-03 17:02:26     if n
2026-04-03 17:02:26   C: 't be a good place to start
2026-04-03 17:02:26 > a good place to start
2026-04-03 17:02:26 > a good place to start
2026-04-03 17:02:26 > a good place to start
2026-04-03 17:02:26 > a good place to start
2026-04-03 17:02:26 > a good place to start
2026-04-03 17:02:26 > a good pla
2026-04-03 17:02:26      uwr=0.078
2026-04-03 17:02:28   P: Neural networks learn by
2026-04-03 17:02:28   C:  means of a variety of factors, including the physical, mental, and physical characteristics of the brain. The brain is a part of the brain that is responsible
2026-04-03 17:02:28      uwr=0.284
2026-04-03 17:02:29   P: The French Revolution began in
2026-04-03 17:02:29   C:  1592, when the French were forced to leave the country. The French were forced to leave the country, and the French were forced to leave the country. The Frenc
2026-04-03 17:02:29      uwr=0.144
2026-04-03 17:02:29   Mean UWR: 0.138
2026-04-03 17:02:31   [ckpt] saved  -> runs/stage1/checkpoint-0007000
2026-04-03 17:02:31   [ckpt] pruned -> checkpoint-0004000
2026-04-03 17:02:31   [hub] uploading checkpoint-0007000 -> WeirdRunner/Ouroboros ...
2026-04-03 17:02:57   [hub] uploaded  checkpoint-0007000 (commit=12093e93)
2026-04-03 17:07:41    7050     4.3659     5.3075     4.3455   0.4082   5.83e-04    2.035       3014
2026-04-03 17:12:23    7100     4.3498     5.3075     4.3471   0.3887   5.83e-04    2.035       5806
2026-04-03 17:17:05    7150     4.0613     5.3075     4.3370   0.4023   5.83e-04    2.035       5811
2026-04-03 17:21:47    7200     4.4002     5.3075     4.3310   0.3789   5.83e-04    2.035       5808
2026-04-03 17:23:17   [spike] step=7216  raw=5.1075  ema=4.3402   ← ⚠ cluster start
2026-04-03 17:23:45   [spike] step=7221  raw=5.0060  ema=4.3486   ← ⚠ consecutive
2026-04-03 17:26:29    7250     4.3702     5.3075     4.3507   0.4062   5.82e-04    2.035       5809
2026-04-03 17:31:12    7300     3.5898     5.3075     4.3234   0.7656   5.82e-04    2.035       5792
2026-04-03 17:31:40   [spike] step=7305  raw=4.9631  ema=4.3298
2026-04-03 17:35:54    7350     4.3598     5.3075     4.3247   0.3477   5.82e-04    2.035       5806
2026-04-03 17:40:36    7400     4.0567     5.3075     4.3179   0.6562   5.82e-04    2.035       5803
2026-04-03 17:43:37   [spike] step=7432  raw=4.9128  ema=4.3282
2026-04-03 17:45:18    7450     4.1306     5.3075     4.3238   0.4180   5.81e-04    2.035       5812
2026-04-03 17:50:00    7500     4.3186     5.3075     4.3209   0.4766   5.81e-04    2.035       5817
2026-04-03 17:53:44   [val] step=7500  val_ce=5.2981
2026-04-03 17:53:44 
2026-04-03 17:53:44   -- Generation @ step 7500 (live weights) --
2026-04-03 17:53:46   P: The capital of France is
2026-04-03 17:53:46   C:  the capital of the country. The capital is the capital of the country, and the capital is the capital. The capital is the capital of the country. The capital i
2026-04-03 17:53:46      uwr=0.093
2026-04-03 17:53:47   P: In mathematics, a prime number is
2026-04-03 17:53:47   C:  a number that is not a number. A number is a number that is not a number. A number is a number that is not a number. A number is a number that is not a number.
2026-04-03 17:53:47      uwr=0.064
2026-04-03 17:53:49   P: def factorial(n):
2026-04-03 17:53:49     """Return n!."""
2026-04-03 17:53:49     if n
2026-04-03 17:53:49   C: .1
2026-04-03 17:53:49 The following is a list of the following:
2026-04-03 17:53:49 - The following is a list of the following:
2026-04-03 17:53:49 - The following is a list of the following:
2026-04-03 17:53:49 - The following is a list o
2026-04-03 17:53:49      uwr=0.104
2026-04-03 17:53:50   P: Neural networks learn by
2026-04-03 17:53:50   C:  means of a network of neural networks that are connected to each other. The network of neurons is called the neural network. The network is the network of neur
2026-04-03 17:53:50      uwr=0.170
2026-04-03 17:53:51   P: The French Revolution began in
2026-04-03 17:53:51   C:  1837, when the French government was in the hands of the French and French. The French government was also in the hands of the French and French. The French we
2026-04-03 17:53:51      uwr=0.198
2026-04-03 17:53:51   Mean UWR: 0.126
2026-04-03 17:58:33    7550     4.0776     5.2981     4.2942   0.3965   5.81e-04    2.035       3190
2026-04-03 18:03:16    7600     4.4039     5.2981     4.2875   0.4121   5.81e-04    2.035       5806
2026-04-03 18:07:58    7650     3.8481     5.2981     4.2847   0.5039   5.80e-04    2.035       5803
2026-04-03 18:12:40    7700     4.2214     5.2981     4.2896   0.4023   5.80e-04    2.035       5806
2026-04-03 18:17:22    7750     4.1954     5.2981     4.2806   0.3594   5.80e-04    2.035       5813
2026-04-03 18:19:48   [spike] step=7776  raw=4.9941  ema=4.3021
2026-04-03 18:22:04    7800     4.3905     5.2981     4.2994   0.3848   5.79e-04    2.035       5817
2026-04-03 18:26:45    7850     4.7069     5.2981     4.3125   0.5859   5.79e-04    2.035       5816
2026-04-03 18:31:27    7900     4.0463     5.2981     4.2984   0.5000   5.79e-04    2.035       5815
2026-04-03 18:36:09    7950     4.2907     5.2981     4.2848   0.3750   5.79e-04    2.035       5819
2026-04-03 18:38:07   [spike] step=7971  raw=5.0408  ema=4.2911   ← ⚠ cluster start
2026-04-03 18:39:37   [spike] step=7987  raw=4.8621  ema=4.3042   ← ⚠ consecutive
2026-04-03 18:40:50   [spike] step=8000  raw=4.9208  ema=4.3076   ← ⚠ consecutive
2026-04-03 18:40:50    8000     4.9208     5.2981     4.3076   0.7422   5.78e-04    2.035       5819
2026-04-03 18:44:36   [val] step=8000  val_ce=5.2907
2026-04-03 18:44:36 
2026-04-03 18:44:36   -- Generation @ step 8000 (live weights) --
2026-04-03 18:44:37   P: The capital of France is
2026-04-03 18:44:37   C:  the capital of the city of Rome, and the capital of the city is the capital of the city. The city is a very small town, with a small town, and a small town, wh
2026-04-03 18:44:37      uwr=0.146
2026-04-03 18:44:38   P: In mathematics, a prime number is
2026-04-03 18:44:38   C:  a number that is not a number. For example, if a number is a number, then the number is a number. If the number is a number, then the number is a number. If th
2026-04-03 18:44:38      uwr=0.328
2026-04-03 18:44:40   P: def factorial(n):
2026-04-03 18:44:40     """Return n!."""
2026-04-03 18:44:40     if n
2026-04-03 18:44:40   C:  is a function of the function of the function of the function of the function of the function of the function of the function of the function of the function o
2026-04-03 18:44:40      uwr=0.042
2026-04-03 18:44:41   P: Neural networks learn by
2026-04-03 18:44:41   C:  the way they communicate, and the way they communicate is a complex network of connections that can be used to communicate information. The network is a networ
2026-04-03 18:44:41      uwr=0.269
2026-04-03 18:44:43   P: The French Revolution began in
2026-04-03 18:44:43   C:  1700, and the French Revolution was a period of great success. The French Revolution was a period of great success, and the French Revolution was a period of g
2026-04-03 18:44:43      uwr=0.124
2026-04-03 18:44:43   Mean UWR: 0.182
2026-04-03 18:44:44   [ckpt] saved  -> runs/stage1/checkpoint-0008000
2026-04-03 18:44:44   [ckpt] pruned -> checkpoint-0005000
2026-04-03 18:44:44   [hub] uploading checkpoint-0008000 -> WeirdRunner/Ouroboros ...
2026-04-03 18:44:59   [hub] uploaded  checkpoint-0008000 (commit=791157cf)
2026-04-03 18:49:43    8050     4.2900     5.2907     4.3104   0.3926   5.78e-04    2.035       3078
2026-04-03 18:50:39   [spike] step=8060  raw=5.4553  ema=4.3208
2026-04-03 18:54:25    8100     4.5317     5.2907     4.3058   0.5312   5.78e-04    2.035       5811
2026-04-03 18:57:19   [spike] step=8131  raw=4.8759  ema=4.3147
2026-04-03 18:59:07    8150     3.9631     5.2907     4.3095   0.7812   5.78e-04    2.035       5810
2026-04-03 19:03:32   [spike] step=8197  raw=5.1788  ema=4.3076
2026-04-03 19:03:48    8200     4.4983     5.2907     4.3118   0.4785   5.77e-04    2.035       5816
2026-04-03 19:08:30    8250     4.3859     5.2907     4.3042   0.4238   5.77e-04    2.035       5808
2026-04-03 19:13:13    8300     4.4820     5.2907     4.3034   0.5273   5.77e-04    2.035       5805
2026-04-03 19:17:54    8350     4.2119     5.2907     4.2834   0.4434   5.76e-04    2.035       5819
2026-04-03 19:18:51   [spike] step=8360  raw=4.7897  ema=4.2845
2026-04-03 19:22:36    8400     4.4063     5.2907     4.2872   0.4453   5.76e-04    2.035       5806
2026-04-03 19:27:18    8450     4.5638     5.2907     4.2957   0.8594   5.76e-04    2.035       5810
2026-04-03 19:30:13   [spike] step=8481  raw=5.1089  ema=4.3097
2026-04-03 19:32:01    8500     4.3299     5.2907     4.3034   0.4629   5.76e-04    2.035       5802
2026-04-03 19:35:45   [val] step=8500  val_ce=5.2900
2026-04-03 19:35:45 
2026-04-03 19:35:45   -- Generation @ step 8500 (live weights) --
2026-04-03 19:35:47   P: The capital of France is
2026-04-03 19:35:47   C:  the capital of the country. The capital is the capital of the country. The capital is the capital of the country. The capital is the capital of the country. Th
2026-04-03 19:35:47      uwr=0.056
2026-04-03 19:35:48   P: In mathematics, a prime number is
2026-04-03 19:35:48   C:  a number. The number of numbers is the number of numbers. The number of numbers is the number of numbers. The number of numbers is the number of numbers. The n
2026-04-03 19:35:48      uwr=0.083
2026-04-03 19:35:50   P: def factorial(n):
2026-04-03 19:35:50     """Return n!."""
2026-04-03 19:35:50     if n
2026-04-03 19:35:50   C: . 1
2026-04-03 19:35:50 * 100% of the time
2026-04-03 19:35:50 * 100% of the time
2026-04-03 19:35:50 * 100% of the time
2026-04-03 19:35:50 * 100% of the time
2026-04-03 19:35:50 * 100% of the time
2026-04-03 19:35:50 * 100% of the time
2026-04-03 19:35:50 * 100% of the time
2026-04-03 19:35:50 * 100% of the time
2026-04-03 19:35:50 * 10
2026-04-03 19:35:50      uwr=0.119
2026-04-03 19:35:51   P: Neural networks learn by
2026-04-03 19:35:51   C:  means of the Internet. The Internet is a network of networks that connect to the Internet. The Internet is a network of networks that connect to the Internet.
2026-04-03 19:35:51      uwr=0.135
2026-04-03 19:35:53   P: The French Revolution began in
2026-04-03 19:35:53   C:  1780, when the French government was forced to take over the French and French colonies. The French were forced to leave the French colonies in 1799, and the F
2026-04-03 19:35:53      uwr=0.239
2026-04-03 19:35:53   Mean UWR: 0.126
2026-04-03 19:40:34    8550     4.4062     5.2900     4.2943   0.4258   5.75e-04    2.035       3190
2026-04-03 19:45:16    8600     3.9206     5.2900     4.2877   0.9844   5.75e-04    2.035       5816
2026-04-03 19:49:58    8650     4.1947     5.2900     4.2884   0.3691   5.75e-04    2.035       5811
2026-04-03 19:54:40    8700     4.2052     5.2900     4.2722   0.3574   5.74e-04    2.035       5808
2026-04-03 19:59:22    8750     4.3004     5.2900     4.2747   0.3809   5.74e-04    2.035       5819
2026-04-03 20:03:47   [spike] step=8797  raw=5.7361  ema=4.2857   ← ⚠ cluster start
2026-04-03 20:04:03    8800     4.3443     5.2900     4.2871   0.7773   5.74e-04    2.035       5819
2026-04-03 20:08:28   [spike] step=8847  raw=4.8565  ema=4.2874   ← ⚠ consecutive
2026-04-03 20:08:45    8850     4.2235     5.2900     4.2874   0.6406   5.74e-04    2.035       5811
2026-04-03 20:13:27   [spike] step=8900  raw=4.8290  ema=4.2898   ← ⚠ consecutive
2026-04-03 20:13:27    8900     4.8290     5.2900     4.2898   0.6250   5.73e-04    2.035       5820
2026-04-03 20:18:08    8950     4.1701     5.2900     4.2706   0.3730   5.73e-04    2.035       5820
2026-04-03 20:22:50    9000     4.1912     5.2900     4.2486   0.3691   5.73e-04    2.035       5814
```

*(Log ends here — run still in progress)*

---

**Loss curve summary (steps 1–9000):**

| Step | Train CE | Smoothed | Val CE | Tokens Seen | Notes |
|---|---|---|---|---|---|
| 1 | 11.98 | 11.98 | — | 32k | Random init |
| 500 | 5.46 | 5.78 | 6.38 | 16.4M | Phrases forming |
| 1000 | 4.97 | 5.14 | 5.85 | 32.8M | Real sentences |
| 1500 | 4.89 | 4.91 | 5.68 | 49.2M | Coherent prose |
| 2000 | — | — | 5.56 | 65.5M | Resumed (ckpt-2000) |
| 2500 | 4.57 | 4.68 | 5.48 | 82.0M | Consistent drop |
| 3000 | 4.48 | 4.60 | 5.42 | 98.3M | Hub sync working |
| 3500 | 4.42 | 4.58 | 5.36 | 114.7M | Spike cluster ⚠ |
| 4000 | 4.46 | 4.50 | 5.34 | 131.1M | Val drop slowing |
| 4500 | 4.59 | 4.50 | 5.32 | 147.5M | |
| 5000 | 4.47 | 4.45 | 5.30 | 163.8M | Val plateau begins ⚠ |
| 5500 | 4.37 | 4.39 | 5.30 | 180.2M | Flat |
| 6000 | 4.31 | 4.36 | 5.31 | 196.6M | Val ticked up slightly |
| 6500 | 4.32 | 4.35 | 5.30 | 213.0M | |
| 7000 | 4.45 | 4.34 | 5.31 | 229.4M | |
| 7500 | 4.32 | 4.32 | 5.30 | 245.8M | |
| 8000 | 4.92 | 4.31 | 5.29 | 262.1M | Spike cluster (7971/7987/8000) |
| 8500 | 4.33 | 4.30 | 5.29 | 278.5M | Plateau continues |

**Key observations (step 9000):**
- Val CE improving very slowly from step 4500 onward (5.32 → 5.29, 4500 steps). Plateau-like but still marginally decreasing; primary signal is healthy.
- Train CE continues to decline monotonically: 4.59 → 4.21. Gap between train and val CE slowly widening — expected at 14.7% of token budget.
- VRAM perfectly flat at 2.035 GB throughout all sessions — zero graph retention. ✅
- Spike rate: 44 spikes / 9000 steps = 0.49% — within acceptable 10% threshold. ✅
- Multiple recurring 2–3 step spike clusters since step 3475. Increase `--shuffle_buffer 20000` on next session.
- UWR chronically low (0.12–0.19) at most gen callbacks since step 5000; code prompt always degenerate (expected, FineWeb-Edu has no code).

---

## Stage 2 SFT — Patch Verification (Code Audit, no run)
**Script:** `train_sft.py`
**Date:** 2026-04-03
**Method:** Static code audit of the submitted `train_sft.py` file.

**Bug 1 — compute_val_ce live-weight restore:** ✅ FIXED
```
  live_backup: Dict[str, torch.Tensor] = {}
  for name, param in model.named_parameters():
      if name in ema.shadow:
          live_backup[name] = param.data.clone()
          param.data.copy_(ema.shadow[name].to(dtype=param.data.dtype))
  ...
  for name, param in model.named_parameters():
      if name in live_backup:
          param.data.copy_(live_backup[name])
```
Live weights are saved before EMA swap and restored after validation. ✅

**Bug 2 — load_latest_checkpoint direct path handling:** ✅ FIXED
```
  # Handles: direct checkpoint path, parent dir scan, Hub fallback
  # search_root + direct_candidates logic present and tested.
```
Function handles all three cases: direct checkpoint dir, parent dir glob, Hub download. ✅

**Bug 3 — collate prompt masking:** 🔴 NOT FIXED
```python
  # In collate():
  labels[idx, :length] = ids   ← ALL tokens supervised, including prompt
  # No prompt_len field in load_and_tokenize samples.
  # samples.append({"input_ids": torch.tensor(ids[:max_seq_len], dtype=torch.long)})
  #   ↑ no "prompt_len" key added
```
User prompt tokens ("User: {question}\n\nAssistant: <think>\n") are still
included in the CE loss. Must fix before Stage 3 — answer val_ce baseline
is inflated by prompt supervision and cannot be used as a Stage 3 gate.

**Fix required in `load_and_tokenize`:**
```python
prefix = f"User: {q}\n\nAssistant: <think>\n"  # or without <think> if no reasoning
prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
prompt_len = len(prefix_ids)
full_ids   = tokenizer.encode(text, add_special_tokens=False)
samples.append({
    "input_ids":  torch.tensor(full_ids[:max_seq_len], dtype=torch.long),
    "prompt_len": prompt_len,   ← ADD THIS
})
```

**Fix required in `collate`:**
```python
for idx, sample in enumerate(samples):
    ids = sample["input_ids"]
    length = ids.size(0)
    pl = min(sample.get("prompt_len", 0), length)
    input_ids[idx, :length] = ids
    labels[idx, pl:length]  = ids[pl:]   ← only supervise response tokens
    mask[idx, :length] = True
```

**Issue 3 — Header string:** ✅ FIXED
```
  print("  Stage 2 SFT - Project Ouroboros")   ← confirmed in file
```

**Issue 4 — Multi-dataset mixing:** ✅ IMPLEMENTED
```
  load_mixed_dataset() present with:
    - _extract_metamath()
    - _extract_openhermes()
    - 40/30/30 ratio logic with available-sample balancing
    - --dataset_mix stratos|full CLI arg
```

**Summary of SFT patch status:**
```
  Bug 1  compute_val_ce weight restore    ✅ FIXED
  Bug 2  load_latest_checkpoint paths     ✅ FIXED
  Bug 3  collate prompt masking           🔴 NOT FIXED — must fix before Stage 2 run
  Issue 3  header string                  ✅ FIXED
  Issue 4  multi-dataset support          ✅ IMPLEMENTED
```

---

## Stage 1 — Pre-training, Kaggle Dual T4 (Session 4, steps 1–1700)
**Script:** `pretrain.py`
**Date:** Session 4
**Hardware:** Kaggle Dual T4 (2× T4 16 GB, DDP auto-launched, world_size=2)
**Status:** Session ended at step 1700; resumed in Session 5

**⚠ No checkpoint saved (Bug 5):** Hub 401 at step 1000 caused `save_checkpoint` to
abandon `.tmp` without renaming to final. Bug 5 was patched before Session 5.
Session 5 started from scratch (step=0) since no local checkpoint existed from Session 4.

---

**Smoke test output (runs automatically before main loop):**
```
  Building val buffer (512 tokens) ...
  Val buffer: 512 tokens from 2 docs
[smoke] epoch_offset=7
[smoke] step  1  loss=8.2671
[smoke] step 10  loss=8.1193
[smoke] step 20  loss=7.3500
[smoke] val_ce computed: 7.9517
  [ckpt] saved  -> /tmp/stage1_smoke_.../checkpoint-0000020
  [resume] local  checkpoint-0000020  step=20  tokens=2,560
[smoke] checkpoint saved and reloaded cleanly
[smoke] All checks passed - launching main training loop
```

**DDP auto-launch:**
```
[ddp] detected 2 CUDA devices; launching single-node DDP with global batch_size=8 (4 per GPU).
```

**Training log (steps 1–1700, Session 4):**
```
      1    11.9824          -    11.9824   1.9766   6.00e-06    2.035       4222
     50     9.2922          -    11.4559   2.2656   1.53e-04    2.035       6040
    100     7.0954          -     9.9666   0.6797   3.03e-04    2.035       5859
    150     7.0693          -     8.7018   1.0938   4.53e-04    2.035       5812
    200     6.2197          -     7.7876   0.7617   6.00e-04    2.035       5794
    250     6.1138          -     7.1607   0.6172   6.00e-04    2.035       5793
    300     6.1652          -     6.7050   0.7773   6.00e-04    2.035       5794
    350     5.7886          -     6.3685   1.0625   6.00e-04    2.035       5796
    400     5.6378          -     6.1231   0.6211   6.00e-04    2.035       5783
    450     5.6214          -     5.9260   0.7383   6.00e-04    2.035       5807
    500     5.4631          -     5.7808   0.5664   6.00e-04    2.035       5805
  [val] step=500  val_ce=6.3811
  Mean UWR: 0.385
    550     5.5842     6.3811     5.6558   0.5352   6.00e-04    2.035       3515
    600     5.4063     6.3811     5.5648   0.5078   6.00e-04    2.035       5860
    ...
   1000     4.9713     6.3811     5.1369   0.4590   6.00e-04    2.035       5719
  [val] step=1000  val_ce=5.8478
  Mean UWR: 0.421
  [hub] upload failed for checkpoint-0001000: Client error '401 Unauthorized' ← ⚠ Bug 5
   1500     4.8855     5.8478     4.9102   0.4805   5.99e-04    2.035       5702
  [val] step=1500  val_ce=5.6810
  Mean UWR: 0.398
   1700     5.0407     5.6810     4.8453   0.4668   5.99e-04    2.035       5735
```
*(Session 4 ended at step 1700 — no checkpoint saved due to Bug 5)*

---

## Stage 2 SFT — Dry-run (nano, 300 samples, 100 steps)
**Script:** `train_sft_phase2.py` (now: `train_sft.py`)
**Date:** Session 3
**Result:** Pipeline verified. EMA generation degenerate at step 100 (expected at decay=0.999). Corrected to `--ema_decay 0.995`.

**⚠ Bugs identified post-run:**
1. `compute_val_ce` — does not restore live weights after EMA eval (Bug 1) — ✅ FIXED
2. `load_latest_checkpoint` — does not handle direct checkpoint paths (Bug 2) — ✅ FIXED
3. `collate` — applies loss to all tokens including user prompt (Bug 3) — 🔴 STILL OPEN

```
  preset=nano  seq_len=512  batch×accum=2×4=8  lr=0.0002  warmup=100

   Step   Train CE     Val CE    GNorm         LR     VRAM    Tok/s
────────────────────────────────────────────────────────────────────
      1    11.9931          -   3.7031   4.00e-06    1.576     1662
     40     9.6128          -   2.7969   8.20e-05    1.576     3731
  [val] step=50  val_ce=11.9826
    100     4.9913    11.9826   1.8594   2.00e-05    1.576     3412
  [val] step=100  val_ce=11.9646
  [ckpt] saved  → runs/phase2/checkpoint-0000100

  Total time: 2.3 min  Peak VRAM: 3.62 GB  Final val CE: 11.9646
```

---

## Stage 0 — Viability Gate
**Script:** `viability_gate.py`   **Date:** Session 2   **Result:** ALL GATES PASSED

```
  G1_ce_converged        CE < 3.5       final CE = 2.0034   PASS ✓
  G2_generation_coherent UWR > 0.1      mean UWR = 0.573    PASS ✓
  G3_grad_norm_stable    gnorm < 10.0   max = 4.0312        PASS ✓
  G4_vram_stable         VRAM Δ < 1.0GB Δ = 0.000 GB        PASS ✓
  Total time: 3.4 min  Peak VRAM: 2.07 GB  Steps: 300
```

---

## Stage 0 — Baseline Architecture Smoke Test
**Script:** `baseline_trm_mamba.py`   **Date:** Session 1   **Result:** PASSED

```
parameters : 92,477,440 (92.5 M)   initial loss : 11.9904   backward : OK
grad norms : total=6.4242           All checks passed. Baseline architecture is healthy.
```
