基于 **Python* 的 油管视频下载，仅供测试，适配 Cookie 验证。  

---

## 🔑 环境和功能
- 🐍 需要 **Python 3.9+** 环境
- 🍪 支持 Cookie 验证


---

## 🔗 相关插件与工具

- 📥 导出浏览器 Cookie 插件：[Get cookies.txt locally](https://chromewebstore.google.com/detail/get-cookiestxt-locally/cclelndahbckbenkjhflpdbgdldlbecc)

- 📱 Android 视频下载器：[Seal](https://github.com/JunkFood02/Seal)

- ⚡ 核心下载工具：[yt-dlp](https://github.com/yt-dlp/yt-dlp)

---

## 📦下载测试


### 最好音频
```
yt-dlp -f bestaudio --cookies /home/cookies.txt -o "%(title)s.%(ext)s" "https://www.youtube.com/watch?v=uSuEdw6HAFE"
```

### 最好视频
```
yt-dlp -f "bestvideo+bestaudio/best" --cookies /home/cookies.txt -o "%(title)s.%(ext)s" "https://www.youtube.com/watch?v=uSuEdw6HAFE"
```

