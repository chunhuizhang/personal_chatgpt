# personal_chatgpt

- 语言模型是怎么被训练出来的
    - 1. pre-train：无监督预训练，海量的文本；（学前时代）
    - 2. Alignment (如下的2和3，严格意义上都是 Alignment，都属于对齐技术)
        - 2. SFT：supervised fine-tuning：有监督训练（学生时代），少量有标注；
        - 3. RLHF：真实的人类反馈，强化学习训练； 


## llama 源码阅读

- [llama introduction](https://www.bilibili.com/video/BV1xP411x7TL)
- [llama text/chat completion](https://www.bilibili.com/video/BV1Zu4y1B7gM/)
- [rmsnorm & swiGLU](https://www.bilibili.com/video/BV1e14y1C7G8/)
- [ROPE](https://www.bilibili.com/video/BV1Dh4y1P7KY/)
- [ROPE & apply_rotary_emb](https://www.bilibili.com/video/BV18u411M7j1/)
- [Cache KV](https://www.bilibili.com/video/BV1FB4y1Z79y/)
- [GQA](https://www.bilibili.com/video/BV1vc411o7fa/)
- [Cache kv & Generate process](https://www.bilibili.com/video/BV1Ea4y1d7wx/)
- [Llama3: 最强小模型](https://www.bilibili.com/video/BV15z42167yB/)

## LoRA & PEFT

- [LoRA（Low Rank Adaption）基本原理与基本概念](https://www.bilibili.com/video/BV15T411477N/)
- [LoRA fine-tune 大语言模型](https://www.bilibili.com/video/BV1qz4y1B7LB/)
- [PEFT/LoRA 源码分析](https://www.bilibili.com/video/BV1sV4y1z7uS/)
- [LLaMA，Alpaca LoRA 7B 推理](https://www.bilibili.com/video/BV1Po4y1T7Bn/)
- [peft LoRA merge pipeline（lora inject，svd）](https://www.bilibili.com/video/BV13A4m1w7i6/)

## TRL (Transformer Reinforcement Learning)

- [trl 基础介绍](https://www.bilibili.com/video/BV1zm4y1H79x/)
- [trl reward model 与 RewardTrainer（奖励模型，分类模型）](https://www.bilibili.com/video/BV1GZ421t7oU/)