# ⚡ Transformer 模型大全

Transformer 是一种让电脑“理解、生成和思考”的神经网络结构，  
诞生于 2017 年的论文  
**《Attention is All You Need》**。  
它用“注意力机制（Attention）”取代了传统的循环结构（RNN），  
让 AI 能更快、更聪明地理解语言、图像和声音。

---

## 🧩 一、Transformer 的基本思想

传统神经网络一次只能看“前后几个词”，  
而 Transformer 可以**同时看到整句话的所有词**，  
并决定“哪些词彼此最重要”。

这就像你在读一句话：
> “我昨天吃了一个🍎苹果，它非常甜。”

Transformer 会自动注意到：
> “它” → “苹果” 的关系。  
> “甜” → “苹果” 的特征。  

---

## 🧱 二、Transformer 的三大结构类型

| 类型 | 说明 | 代表模型 | 用途 |
|------|------|-----------|------|
| 🟦 **Encoder-only（只编码）** | 理解输入内容（不生成） | **BERT**, RoBERTa | 文本理解、搜索、分类 |
| 🟥 **Decoder-only（只解码）** | 根据上下文生成新内容 | **GPT 系列**, LLaMA, Claude | 文本生成、对话 |
| 🟨 **Encoder–Decoder（编解码器）** | 既理解又生成 | **T5**, BART, mBART | 翻译、摘要、问答 |

---

## 🧠 三、自然语言处理（NLP）方向的代表模型

### 🟦 Encoder-only 模型（擅长理解）
| 模型 | 发布年份 | 机构 | 主要功能 |
|------|-----------|------|-----------|
| **BERT** | 2018 | Google | 句子理解、分类、问答 |
| **RoBERTa** | 2019 | Meta | BERT 改进版，性能更强 |
| **ALBERT** | 2019 | Google | 参数更少，速度更快 |
| **DeBERTa** | 2021 | Microsoft | 动态注意力改进，效果极好 |
| **ERNIE（文心）** | 2019 | 百度 | 中文语义理解增强 |
| **MacBERT / SimBERT** | 2020 | 哈工大/讯飞 | 中文优化版本 |

---

### 🟥 Decoder-only 模型（擅长生成）
| 模型 | 发布年份 | 开发者 | 特点 |
|------|-----------|----------|----------|
| **GPT (1,2,3,4)** | 2018–2024 | OpenAI | 聊天、写作、推理 |
| **ChatGPT** | 2022 | OpenAI | 基于 GPT-3.5/4 的对话系统 |
| **LLaMA / LLaMA 2 / LLaMA 3** | 2023–2024 | Meta | 开源语言模型系列 |
| **Claude 1–3** | 2023–2024 | Anthropic | 强调安全性与逻辑性 |
| **Gemini（Bard）** | 2023 | Google DeepMind | 文本 + 图片 + 视频多模态 |
| **Mistral / Mixtral** | 2023 | Mistral AI | 高效轻量、性能出色 |
| **Yi / Qwen / Moonshot / 百川 / 讯飞星火** | 2023–2025 | 中国团队 | 中文与多语言强项 |

---

### 🟨 Encoder–Decoder 模型（擅长理解+生成）
| 模型 | 发布年份 | 开发者 | 应用 |
|------|-----------|----------|----------|
| **T5 (Text-to-Text Transfer Transformer)** | 2020 | Google | 所有NLP任务转为文本生成 |
| **mT5 / ByT5** | 2021 | Google | 多语言版 / 字节级版 |
| **BART** | 2019 | Meta | 文本生成、摘要 |
| **mBART** | 2020 | Meta | 多语言翻译、生成 |
| **UL2 / Flan-T5** | 2022 | Google | 通用型生成理解模型 |

---

## 🎨 四、视觉（Vision）方向的 Transformer

| 模型 | 全称 | 发布年份 | 应用 |
|------|------|-----------|------|
| **ViT** | Vision Transformer | 2020 | 图像分类 |
| **Swin Transformer** | Shifted Window Transformer | 2021 | 图像检测、分割 |
| **DETR** | Detection Transformer | 2020 | 目标检测 |
| **DINO / Mask DINO** | - | 2021–2023 | 自监督学习 |
| **SAM (Segment Anything Model)** | 2023 | Meta | 通用图像分割 |
| **CLIP** | Contrastive Language–Image Pretraining | 2021 | 文字与图像对齐（图文理解） |
| **BLIP / BLIP-2** | - | 2022–2023 | 图文问答、生成描述 |

---

## 🔊 五、语音与音频方向

| 模型 | 开发者 | 特点 |
|------|----------|----------|
| **Whisper** | OpenAI | 语音识别 + 多语言翻译 |
| **AudioLM / MusicLM** | Google | 音频、音乐生成 |
| **SpeechT5** | Microsoft | 语音合成与识别统一框架 |
| **Valle / FireRedTTS / IndexTTS** | 多机构 | 基于 Transformer 的语音合成模型 |
| **VITS / FastSpeech 2** | - | 高自然度的 TTS 模型 |

---

## 🎬 六、多模态（Multimodal）Transformer

> 同时理解 **文字 + 图像 + 音频 + 视频**

| 模型 | 发布机构 | 功能 |
|------|-----------|----------|
| **CLIP** | OpenAI | 图文配对（理解图像内容） |
| **ALIGN** | Google | 图文对齐，CLIP 竞争者 |
| **BLIP / BLIP-2** | Salesforce | 图文问答、生成描述 |
| **Flamingo** | DeepMind | 文本+图像多模态对话 |
| **Kosmos-1 / Kosmos-2** | Microsoft | 通用多模态智能体 |
| **Gemini 1 / 1.5 / 2** | Google | 文本+图像+视频+音频理解 |
| **GPT-4 / GPT-4o** | OpenAI | 多模态生成（图文音视频） |
| **Sora** | OpenAI | 文本→视频生成 |
| **Pika / Runway Gen-2 / Kling / Vidu** | 多机构 | AI视频生成Transformer |
| **LLaVA / Qwen-VL / InternVL / Yi-VL** | 中国团队 | 中文多模态理解 |

---

## 🧬 七、科学与专业应用方向

| 模型 | 应用领域 | 说明 |
|------|------------|-----------|
| **AlphaFold / AlphaFold 2** | 生物结构 | 蛋白质折叠预测 |
| **Graphormer / Molformer** | 化学建模 | 分子结构理解 |
| **Galactica / SciBERT / BioBERT** | 学术研究 | 科学论文理解 |
| **CodeBERT / Codex / StarCoder** | 编程领域 | 代码生成与理解 |
| **Med-PaLM / BioGPT** | 医疗领域 | 医学问答与报告生成 |

---

## 📜 八、Transformer 模型发展时间线

| 年份 | 关键模型 | 重大意义 |
|------|-----------|-----------|
| 2017 | Transformer（Attention is All You Need） | 原始论文诞生 |
| 2018 | BERT、GPT-1 | NLP 预训练革命 |
| 2019 | GPT-2、XLNet、BART | 大模型生成初见威力 |
| 2020 | T5、ViT、DETR | 文本与视觉融合 |
| 2021 | CLIP、DALL·E、Whisper、Codex | 多模态与代码理解 |
| 2022 | GPT-3.5、Stable Diffusion、Flan-T5 | 生成式AI爆发 |
| 2023 | GPT-4、Claude、Gemini、SAM、LLaMA | 通用大模型时代 |
| 2024 | GPT-4o、Gemini 1.5、Yi-1.5、Qwen2-VL、Sora | 多模态统一智能 |
| 2025 | Gemini 2、Claude 4、OpenAI Video Model | 视频级通用智能时代开启 |

---

## 💬 九、一句话总结

> Transformer 是现代 AI 的“万能大脑”。  
> 从文字到图片，从语音到视频，从代码到药物设计，  
> 几乎所有强大的智能系统——都是 Transformer 的后代。⚡

---

## 🎯 十、比喻理解（适合学生）

| 模型类型 | 像什么 | 能做什么 |
|-----------|---------|------------|
| **BERT** | 理解大师 🧠 | 看懂文章、分析句子 |
| **GPT** | 作家 ✍️ | 写作文、编故事、对话 |
| **CLIP** | 翻译官 👁️💬 | 看图说话 |
| **ViT** | 摄影师 📷 | 看图识物 |
| **Whisper** | 听力高手 👂 | 听懂多语言语音 |
| **Sora** | 导演 🎬 | 从文字拍视频 |
| **AlphaFold** | 科学家 🔬 | 预测蛋白质结构 |

---

## 📚 十一、总结图（一句话记忆）

> **Transformer 家族：**
> - 🧠 理解：BERT  
> - ✍️ 生成：GPT  
> - 🖼️ 看图：ViT / CLIP  
> - 🎨 画画：DALL·E / Stable Diffusion  
> - 🎧 听音：Whisper  
> - 🎬 拍片：Sora  
> - 💻 编程：Codex / StarCoder  
> - 🔬 科研：AlphaFold / Galactica  

---

✨ **结论：**
> Transformer 已经成为「AI 的通用语言」。  
> 未来的每一种智能，都可能基于 Transformer。
