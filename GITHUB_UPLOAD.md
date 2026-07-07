# GitHub Upload Guide

本目录已整理为可上传 GitHub 的协作仓库。

## 方法 A：已有 GitHub 空仓库

假设仓库地址为：

```text
https://github.com/<org-or-user>/<repo>.git
```

在本目录运行：

```powershell
cd C:\Users\Administrator\Documents\research\atc-hmi-bluesky-visual-collab
git init
git add .
git commit -m "Initial ATC HMI BlueSky visual collaboration package"
git branch -M main
git remote add origin https://github.com/<org-or-user>/<repo>.git
git push -u origin main
```

## 方法 B：还没有 GitHub 仓库

1. 在 GitHub 网页新建仓库，例如：`atc-hmi-bluesky-visual-collab`。
2. 不要勾选初始化 README / .gitignore / license。
3. 回到本地按方法 A 执行。

## 当前本机情况

- 已安装 `git`。
- 未检测到 GitHub CLI `gh`，所以不能直接用命令自动创建 GitHub 仓库。
- 如需我直接 push，请提供 GitHub 仓库 URL，并确保当前终端已有 GitHub 认证权限。
