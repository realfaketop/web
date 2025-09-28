# IndexTTS2 要点整理

## 📅 发布时间
- **2025/09/08** IndexTTS-2 发布（首个支持精确合成时长控制的自回归零样本文本转语音模型）  
- **2025/05/14** IndexTTS-1.5 发布（提升模型稳定性和英文表现）  
- **2025/03/25** IndexTTS-1.0 发布（开放权重和推理代码）  
- **2025/02/12** 论文提交至 arXiv，并发布 Demo 与测试集  

---

## 🤖 模型下载

| **HuggingFace**                                          | **ModelScope** |
|----------------------------------------------------------|----------------------------------------------------------|
| [IndexTTS-2](https://huggingface.co/IndexTeam/IndexTTS-2) | [IndexTTS-2](https://modelscope.cn/models/IndexTeam/IndexTTS-2) |
| [IndexTTS-1.5](https://huggingface.co/IndexTeam/IndexTTS-1.5) | [IndexTTS-1.5](https://modelscope.cn/models/IndexTeam/IndexTTS-1.5) |
| [IndexTTS](https://huggingface.co/IndexTeam/Index-TTS) | [IndexTTS](https://modelscope.cn/models/IndexTeam/Index-TTS) |

---

<div style="text-align:left">
  <a href='https://arxiv.org/abs/2506.21619'>
    <img src='https://img.shields.io/badge/ArXiv-2506.21619-red?logo=arxiv'/>
  </a>
  <br/>
  <a href='https://github.com/index-tts/index-tts'>
    <img src='https://img.shields.io/badge/GitHub-Code-orange?logo=github'/>
  </a>
  <a href='https://index-tts.github.io/index-tts2.github.io/'>
    <img src='https://img.shields.io/badge/GitHub-Demo-orange?logo=github'/>
  </a>
  <br/>
  <a href='https://huggingface.co/spaces/IndexTeam/IndexTTS-2-Demo'>
    <img src='https://img.shields.io/badge/HuggingFace-Demo-blue?logo=huggingface'/>
  </a>
  <a href='https://huggingface.co/IndexTeam/IndexTTS-2'>
    <img src='https://img.shields.io/badge/HuggingFace-Model-blue?logo=huggingface' />
  </a>
  <br/>
  <a href='https://modelscope.cn/studios/IndexTeam/IndexTTS-2-Demo'>
    <img src='https://img.shields.io/badge/ModelScope-Demo-purple?logo=modelscope'/>
  </>
  <a href='https://modelscope.cn/models/IndexTeam/IndexTTS-2'>
    <img src='https://img.shields.io/badge/ModelScope-Model-purple?logo=modelscope'/>
  </a>
</div>

## 📄 论文

* **IndexTTS2**

  > Zhou, Siyi; Zhou, Yiquan; He, Yi; Zhou, Xun; Wang, Jinchao; Deng, Wei; Shu, Jingchen
  > *IndexTTS2: A Breakthrough in Emotionally Expressive and Duration-Controlled Auto-Regressive Zero-Shot Text-to-Speech*
  > arXiv preprint arXiv:2506.21619 (2025)
  > [[Paper Link](https://arxiv.org/abs/2506.21619)](https://arxiv.org/abs/2506.21619)

* **IndexTTS1**

  > Deng, Wei; Zhou, Siyi; Shu, Jingchen; Wang, Jinchao; Wang, Lu
  > *IndexTTS: An Industrial-Level Controllable and Efficient Zero-Shot Text-To-Speech System*
  > arXiv preprint arXiv:2502.05512 (2025)
  > [[Paper Link](https://arxiv.org/abs/2502.05512)](https://arxiv.org/abs/2502.05512)



## ⚙️ 部署方法

### 1. 安装依赖

```bash
# 安装 uv (推荐的依赖管理工具)
pip install -U uv

# 克隆项目
git clone https://github.com/index-tts/index-tts.git && cd index-tts
git lfs install
git lfs pull

# 同步依赖
uv sync --all-extras
# 如果网络慢，可用国内镜像：
uv sync --all-extras --default-index "https://mirrors.aliyun.com/pypi/simple"
```

### 2. 下载模型

```bash
# HuggingFace
uv tool install "huggingface-hub[cli,hf_xet]"
hf download IndexTeam/IndexTTS-2 --local-dir=checkpoints

# 或 ModelScope
uv tool install "modelscope"
modelscope download --model IndexTeam/IndexTTS-2 --local_dir checkpoints
```

### 3. 检查 GPU 环境

```bash
uv run tools/gpu_check.py
```

### 4. 启动 WebUI

```bash
uv run webui.py
# 浏览器访问 http://127.0.0.1:7860
```

### 5. Python 调用示例

```python
from indextts.infer_v2 import IndexTTS2

tts = IndexTTS2(cfg_path="checkpoints/config.yaml", model_dir="checkpoints", use_fp16=True)
text = "Translate for me, what is a surprise!"
tts.infer(spk_audio_prompt='examples/voice_01.wav', text=text, output_path="gen.wav", verbose=True)
```


## 📢 社区与支持

* QQ 群：553460296 (No.1) / 663272642 (No.4)
* Discord：[[Join](https://discord.gg/uT32E7KDmy)](https://discord.gg/uT32E7KDmy)
* Email：[[indexspeech@bilibili.com](mailto:indexspeech@bilibili.com)](mailto:indexspeech@bilibili.com)
* 官方仓库：[https://github.com/index-tts/index-tts](https://github.com/index-tts/index-tts)


## 运行代码

```
# .vscode/preview.yml
autoOpen: true # 打开工作空间时是否自动开启所有应用的预览
apps:
  - port: 7860 # 应用的端口
    run: uv run webui.py
    root: ./index-tts # 应用的启动目录
    name: IndexTTS2  # 应用名称
    description: IndexTTS2 # 应用描述
    autoOpen: true # 打开工作空间时是否自动运行命令（优先级高于根级 autoOpen）
    autoPreview: true # 是否自动打开预览, 若无则默认为true
```