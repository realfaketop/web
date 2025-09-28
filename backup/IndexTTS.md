## 版本更新


- **2025/09/08** IndexTTS-2 发布（首个支持精确合成时长控制的自回归零样本文本转语音模型）  
- **2025/05/14** IndexTTS-1.5 发布（提升模型稳定性和英文表现）  
- **2025/03/25** IndexTTS-1.0 发布（开放权重和推理代码）  
- **2025/02/12** 论文提交至 arXiv，并发布 Demo 与测试集  

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

---

## 模型下载

| **HuggingFace**                                          | **ModelScope** |
|----------------------------------------------------------|----------------------------------------------------------|
| [IndexTTS-2](https://huggingface.co/IndexTeam/IndexTTS-2) | [IndexTTS-2](https://modelscope.cn/models/IndexTeam/IndexTTS-2) |
| [IndexTTS-1.5](https://huggingface.co/IndexTeam/IndexTTS-1.5) | [IndexTTS-1.5](https://modelscope.cn/models/IndexTeam/IndexTTS-1.5) |
| [IndexTTS](https://huggingface.co/IndexTeam/Index-TTS) | [IndexTTS](https://modelscope.cn/models/IndexTeam/Index-TTS) |

---


##  社区支持

* QQ 群：553460296 (No.1) / 663272642 (No.4)
* Discord：[Join](https://discord.gg/uT32E7KDmy)
* Email：[indexspeech@bilibili.com](mailto:indexspeech@bilibili.com)
* 官方仓库：[https://github.com/index-tts/index-tts](https://github.com/index-tts/index-tts)

---

## 论文发布

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


---

## 论文概览

- **研究问题**  
  - 自动回归 TTS：声音自然但时长难控  
  - 非自回归 TTS：时长可控但自然度和情感不足  

- **核心贡献（IndexTTS2）**  
  - 在自动回归框架中首次实现 **精确时长控制**  
  - **声音身份与情感解耦**，可独立控制  
  - 支持 **严格时长模式** 与 **自由模式**  
  - 强化 **情感建模**，高强度情绪仍稳定  
  - 输入灵活：情感可用文本提示指定  

- **实验结果**  
  - 自然度（MOS）优于主流模型  
  - 时长控制更精确，适合对口型场景  
  - 情感表现力更强，避免高情绪失真  
  - 零样本能力突出，能泛化到新声音/情绪  

- **应用场景**  
  - 影视/动画配音  
  - 游戏角色语音  
  - 个性化语音助手  

- **未来方向**  
  - 更大规模情感数据集  
  - 多模态条件控制  

---

## 相关模型

| 名称 | 来源 | 类型 | 发布时间 | 用途 / 备注 |
|------|------|------|----------|-------------|
| MaskGCT | [GitHub](https://github.com/modelscope/MaskGCT) | 零样本文本转语音 (Zero-shot TTS) | 2024 | 作为零样本基线模型，对比 IndexTTS2 在语音质量与时长控制上的表现 |
| F5-TTS | [GitHub](https://github.com/SWivid/F5-TTS) | 零样本 TTS / Diffusion 模型 | 2024 | 提供高自然度语音生成，对比 IndexTTS2 的情感与时长控制 |
| CosyVoice2 | [HuggingFace](https://huggingface.co/netease-youdao/CosyVoice2) | 多语种 / 零样本 TTS | 2025 | 在跨语言和情感表达方面作为对比模型 |
| SparkTTS | [GitHub](https://github.com/Lightning-AI/spark-tts) | 零样本 / 高速 TTS | 2024 | 强调快速推理与低延迟，对比 IndexTTS2 在自然度与灵活性 |


## 相关数据

| 名称 | 来源 | 类型 | 发布时间 | 用途 / 备注 |
|------|------|------|----------|-------------|
| ESD (Emotional Speech Dataset) | [GitHub](https://github.com/HLTSingapore/ESD) | 多语言情感语音数据集 | 2020 | 提供 29 小时情感语音，用于增强情感建模 |
| Common Voice | [Mozilla Common Voice](https://commonvoice.mozilla.org/) | 多语言众包语音 | 2017 | 部分语料被采样用于 SeedTTS test-en 基准测试 |
| DiDiSpeech | [HuggingFace](https://huggingface.co/datasets/didi-ai/DiDiSpeech) | 中文普通话语音 | 2021 | 部分语料被采样用于 SeedTTS test-zh 基准测试 |
| LibriSpeech | [OpenSLR](https://www.openslr.org/12/) | 英语朗读语音 (有声书) | 2015 | 随机抽取 test-clean 子集，用于英语语音评测 |
| AISHELL-1 | [OpenSLR](https://www.openslr.org/33/) | 中文普通话语音 | 2017 | 随机抽取 1,000 条语音作为测试集 |

---

##  部署方法

#### 1. 安装依赖

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

#### 2. 下载模型

```bash
# HuggingFace
uv tool install "huggingface-hub[cli,hf_xet]"
hf download IndexTeam/IndexTTS-2 --local-dir=checkpoints

# 或 ModelScope
uv tool install "modelscope"
modelscope download --model IndexTeam/IndexTTS-2 --local_dir checkpoints
```

#### 3. 检查 GPU 环境

```bash
uv run tools/gpu_check.py
```

#### 4. 启动 WebUI

```bash
uv run webui.py
# 浏览器访问 http://127.0.0.1:7860
```

#### 5. Python 调用示例

```python
from indextts.infer_v2 import IndexTTS2

tts = IndexTTS2(cfg_path="checkpoints/config.yaml", model_dir="checkpoints", use_fp16=True)
text = "Translate for me, what is a surprise!"
tts.infer(spk_audio_prompt='examples/voice_01.wav', text=text, output_path="gen.wav", verbose=True)
```

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

## 批量生成

```
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
IndexTTS2 批量生成（离线/脚本版）
- 自动安装依赖（含 modelscope / transformers / accelerate / WeTextProcessing / descript-audiotools）
- 读取 in/1.txt（每行一句）和 in/ 下全部参考音频（wav/mp3/flac/m4a/ogg）
- 做“音频 × 文本”笛卡尔积，生成到 out/
默认路径：
  --in_dir    /workspace/index-tts/in
  --out_dir   /workspace/index-tts/out
  --model_dir /workspace/index-tts/checkpoints
"""
import os, sys, time, subprocess, importlib
from pathlib import Path
import argparse

# -------------------- 依赖自动安装（含版本对齐 & 特殊回退） --------------------
PINNED_PKGS = {
    # 基础
    "numpy": "numpy>=1.24",
    "scipy": "scipy>=1.10",
    "soundfile": "soundfile>=0.12",
    "librosa": "librosa>=0.10",
    "einops": "einops>=0.6",
    "sentencepiece": "sentencepiece>=0.1.99",
    "safetensors": "safetensors>=0.4.2",
    "tqdm": "tqdm>=4.66",
    "packaging": "packaging>=23.2",
    "omegaconf": "omegaconf>=2.3.0",
    # 关键三件套
    "transformers": "transformers>=4.44.2",
    "accelerate": "accelerate>=0.26.0",
    "modelscope": "modelscope>=1.19.0",
    # DAC 依赖（导入名 audiotools；特殊逻辑见下）
    "descript-audiotools": "descript-audiotools",
    # 常见封装辅助（可选）
    "av": "av>=12.0.0",
    "ffmpeg-python": "ffmpeg-python>=0.2.0",
    # 中文文本正则化（提供 tn.chinese.normalizer）
    "WeTextProcessing": "WeTextProcessing",
}

IMPORT_NAME_MAP = {
    "descript-audiotools": "audiotools",
    "ffmpeg-python": "ffmpeg",
    "WeTextProcessing": "tn",   # 安装包名 WeTextProcessing，导入名 tn
}

ALIYUN = "https://mirrors.aliyun.com/pypi/simple/"
PYPI = "https://pypi.org/simple/"

def _import_name_of(spec_key: str) -> str:
    return IMPORT_NAME_MAP.get(spec_key, spec_key.split("[")[0].split("==")[0].split(">=")[0])

def pip_install(args_list):
    print("[PIP]", " ".join(map(str, args_list)))
    subprocess.check_call(args_list)

def install_general(spec_key: str, spec_value: str):
    """通用安装：默认索引 -> 阿里镜像"""
    mod = _import_name_of(spec_key)
    try:
        importlib.import_module(mod)
        return
    except Exception:
        pass
    try:
        pip_install([sys.executable, "-m", "pip", "install", "-U", spec_value])
        importlib.import_module(mod); return
    except Exception as e1:
        print(f"[WARN] install {spec_value} on default index failed: {e1}")
    try:
        pip_install([sys.executable, "-m", "pip", "install", "-U", spec_value, "-i", ALIYUN])
        importlib.import_module(mod); return
    except Exception as e2:
        print(f"[WARN] install {spec_value} on Aliyun failed: {e2}")
        raise

def install_descript_audiotools():
    """descript-audiotools 特殊处理：
       1) PyPI 最新；2) PyPI 0.7.2；3) 阿里 0.7.2
    """
    mod = "audiotools"
    try:
        importlib.import_module(mod); return
    except Exception:
        pass
    for spec, idx in [
        ("descript-audiotools", PYPI),
        ("descript-audiotools==0.7.2", PYPI),
        ("descript-audiotools==0.7.2", ALIYUN),
    ]:
        try:
            pip_install([sys.executable, "-m", "pip", "install", "-U", spec, "-i", idx])
            importlib.import_module(mod); return
        except Exception as e:
            print(f"[WARN] {spec} via {idx} failed: {e}")
    raise RuntimeError("Failed to install descript-audiotools")

def install_wetextprocessing():
    """WeTextProcessing 提供 tn.*：
       1) PyPI 最新；2) 阿里最新；失败则报错
    """
    try:
        importlib.import_module("tn"); return
    except Exception:
        pass
    for idx in [PYPI, ALIYUN]:
        try:
            pip_install([sys.executable, "-m", "pip", "install", "-U", "WeTextProcessing", "-i", idx])
            importlib.import_module("tn"); return
        except Exception as e:
            print(f"[WARN] WeTextProcessing via {idx} failed: {e}")
    raise RuntimeError("Failed to install WeTextProcessing (tn)")

def ensure_deps():
    for k, v in PINNED_PKGS.items():
        if k == "descript-audiotools":
            install_descript_audiotools()
        elif k == "WeTextProcessing":
            install_wetextprocessing()
        else:
            install_general(k, v)
    # 关键：立刻验证 accelerate
    try:
        import accelerate  # noqa: F401
    except Exception as e:
        print("[FATAL] accelerate 仍不可用：", e)
        print("请退出当前进程后重跑本脚本；或手动执行：")
        print("  python -m pip install -U 'transformers>=4.44.2' 'accelerate>=0.26.0' 'modelscope>=1.19.0'")
        sys.exit(1)

ensure_deps()

# -------------------- 业务逻辑 --------------------
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.append(str(SCRIPT_DIR))
sys.path.append(str(SCRIPT_DIR / "indextts"))

from indextts.infer_v2 import IndexTTS2  # noqa: E402

def find_prompt_audios(in_dir: Path):
    exts = ["*.wav", "*.mp3", "*.flac", "*.m4a", "*.ogg"]
    files = []
    for p in exts: files += list(in_dir.glob(p))
    return sorted([p for p in files if p.is_file()], key=lambda x: x.name.lower())

def read_lines(txt: Path):
    with txt.open("r", encoding="utf-8") as f:
        return [ln.strip() for ln in f if ln.strip()]

def safe_stem(p: Path) -> str:
    return p.stem.replace(" ", "_").replace("/", "_").replace("\\", "_")[:80]

def detect_fp16() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False

def main():
    parser = argparse.ArgumentParser(description="IndexTTS2 批量生成（音频 × 文本）")
    parser.add_argument("--in_dir", type=str, default="/workspace/index-tts/in", help="输入目录（含 1.txt 与参考音频）")
    parser.add_argument("--out_dir", type=str, default="/workspace/index-tts/out", help="输出目录")
    parser.add_argument("--model_dir", type=str, default="/workspace/index-tts/checkpoints", help="模型目录（含 config.yaml 等）")
    # 生成参数
    parser.add_argument("--max_text_tokens_per_segment", type=int, default=120)
    parser.add_argument("--do_sample", action="store_true", default=True)
    parser.add_argument("--top_p", type=float, default=0.8)
    parser.add_argument("--top_k", type=int, default=30)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--num_beams", type=int, default=3)
    parser.add_argument("--repetition_penalty", type=float, default=10.0)
    parser.add_argument("--max_mel_tokens", type=int, default=1500)
    args = parser.parse_args()

    in_dir = Path(args.in_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    model_dir = Path(args.model_dir).resolve()

    if not in_dir.exists():
        print(f"[ERROR] 输入目录不存在：{in_dir}"); sys.exit(1)
    txt = in_dir / "1.txt"
    if not txt.exists():
        print(f"[ERROR] 缺少文本文件：{txt}"); sys.exit(1)
    if not model_dir.exists():
        print(f"[ERROR] 模型目录不存在：{model_dir}"); sys.exit(1)

    prompts = find_prompt_audios(in_dir)
    texts = read_lines(txt)
    if not prompts:
        print(f"[ERROR] {in_dir} 下未找到音频（支持 wav/mp3/flac/m4a/ogg）"); sys.exit(1)
    if not texts:
        print(f"[ERROR] {txt} 为空"); sys.exit(1)

    out_dir.mkdir(parents=True, exist_ok=True)

    use_fp16 = detect_fp16()
    print(f"[INFO] Loading IndexTTS2 ... (fp16={use_fp16})")
    tts = IndexTTS2(
        cfg_path=str(model_dir / "config.yaml"),
        model_dir=str(model_dir),
        use_fp16=use_fp16,
        use_cuda_kernel=False,
        use_deepspeed=False,
    )
    print(f"[INFO] Model loaded. version={tts.model_version or '1.0'}")

    total = len(prompts) * len(texts)
    idx = 0
    for pa in prompts:
        base = safe_stem(pa)
        for li, text in enumerate(texts, 1):
            idx += 1
            ts = int(time.time() * 1000)
            out_path = out_dir / f"{base}__L{li:03d}__{ts}.wav"
            print(f"[{idx}/{total}] {pa.name} × L{li} -> {out_path.name}")
            try:
                tts.infer(
                    spk_audio_prompt=str(pa),
                    text=text,
                    output_path=str(out_path),
                    emo_audio_prompt=None,
                    emo_alpha=0.65,
                    emo_vector=None,
                    use_emo_text=False,
                    emo_text=None,
                    use_random=False,
                    verbose=False,
                    max_text_tokens_per_segment=int(args.max_text_tokens_per_segment),
                    do_sample=bool(args.do_sample),
                    top_p=float(args.top_p),
                    top_k=int(args.top_k) if int(args.top_k) > 0 else None,
                    temperature=float(args.temperature),
                    length_penalty=0.0,
                    num_beams=int(args.num_beams),
                    repetition_penalty=float(args.repetition_penalty),
                    max_mel_tokens=int(args.max_mel_tokens),
                )
            except Exception as e:
                print(f"[ERROR] 生成失败：{e}")

    print(f"[DONE] 共生成 {total} 条音频，输出目录：{out_dir}")

if __name__ == "__main__":
    main()
```