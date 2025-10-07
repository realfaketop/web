基于 **Python** 的 油管视频下载，仅供测试，适配 Cookie 验证。  

---

## 🔑 环境和功能
- 🐍 需要 **Python 3.9+** 环境
- 🍪 支持 Cookie 验证


---

## 🔗 相关插件与工具

- 📥 导出浏览器 Cookie 插件：[Get cookies.txt locally](https://chromewebstore.google.com/detail/get-cookiestxt-locally/cclelndahbckbenkjhflpdbgdldlbecc)

- 📱 Android 视频下载器：[Seal](https://github.com/JunkFood02/Seal)
- ⚡ 音频字幕生成：[srt](https://github.com/WEIFENG2333/VideoCaptioner)
- ⚡ 核心下载工具：[yt-dlp](https://github.com/yt-dlp/yt-dlp)
-  ⚡Google ipynb  [colab](https://colab.research.google.com/drive/1wnFybq6zJkF3w4OE2AKs54HV0RUB4leM?usp=sharing)

---

## 📦下载测试


### 最好音频
```
yt-dlp -f bestaudio --cookies /home/cookies.txt -o "%(title)s.%(ext)s" "https://www.youtube.com/watch?v=uSuEdw6HAFE"
```

### 最好音频视频
```
yt-dlp -f "bestvideo+bestaudio/best" --cookies /home/cookies.txt -o "%(title)s.%(ext)s" "https://www.youtube.com/watch?v=uSuEdw6HAFE"
```

## 字幕视频合并

```
# =====================================================
# 🎬 一键无损合并 /content/123.webm + /content/123666.srt
# 自动输出两种版本：MP4（通用） & MKV（原封装）
# =====================================================

import subprocess

# 输入文件路径
video_path = "/content/123.webm"
srt_path   = "/content/123666.srt"

# 输出文件路径
output_mp4 = "/content/123_merged.mp4"
output_mkv = "/content/123_merged.mkv"

# 安装 ffmpeg（Colab 一次即可）
!apt -y update -qq && apt -y install ffmpeg -qq

print("🚀 正在无损合并为 MP4 ...")
subprocess.run([
    "ffmpeg", "-i", video_path, "-i", srt_path,
    "-c:v", "copy", "-c:a", "copy", "-c:s", "mov_text",
    "-metadata:s:s:0", "language=chi", "-metadata:s:s:0", "title=中文字幕",
    "-map", "0:v", "-map", "0:a?", "-map", "1:0",
    "-disposition:s:0", "default",
    output_mp4, "-y"
])

print("🚀 正在无损合并为 MKV ...")
subprocess.run([
    "ffmpeg", "-i", video_path, "-i", srt_path,
    "-c", "copy", "-c:s", "srt",
    "-metadata:s:s:0", "language=chi", "-metadata:s:s:0", "title=中文字幕",
    "-map", "0:v", "-map", "0:a?", "-map", "1:0",
    "-disposition:s:0", "default",
    output_mkv, "-y"
])

print("✅ 已生成：")
print("MP4 文件：", output_mp4)
print("MKV 文件：", output_mkv)

from google.colab import files
print("\n📥 可点击以下命令下载文件：")
print("files.download('/content/123_merged.mp4')")
print("files.download('/content/123_merged.mkv')")
```
