# Image Retrieval

一个基于 TensorFlow Inception v3 特征和余弦相似度的以图搜图 Web 应用。用户上传查询图片后，Flask 服务会返回数据集中最相似的 9 张图片。

## 项目结构

```text
.
├── server/
│   ├── app.py                 # Flask 服务入口
│   ├── download_model.py      # 下载 Inception 模型
│   ├── image_vectorizer.py    # 为数据集生成特征索引
│   ├── inception.py           # 模型加载和特征提取
│   ├── search.py              # 相似图片检索
│   ├── database/dataset/      # 本地图片数据集，不提交 Git
│   ├── data/                  # 生成的压缩特征索引，不提交 Git
│   ├── static/                # 页面静态资源
│   └── templates/             # HTML 模板
├── requirements.txt
└── LICENSE
```

大型数据集、模型文件、上传文件和生成的特征索引已加入 `.gitignore`，避免超过 GitHub 文件大小限制。

## 环境要求

- Python 3.9
- 建议至少保留 2 GB 可用磁盘空间

该项目使用 TensorFlow 1.x 兼容接口，并通过 TensorFlow 2.10.1 的 `compat.v1` 运行。

## 安装

```powershell
py -3.9 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## 准备模型

如果本地尚无 `server/imagenet/classify_image_graph_def.pb`：

```powershell
python server/download_model.py
```

## 准备数据集和特征

把 JPEG 或 PNG 图片放入：

```text
server/database/dataset/
```

然后生成压缩特征和图片索引：

```powershell
python server/image_vectorizer.py
```

也可以指定其他数据集或限制处理数量：

```powershell
python server/image_vectorizer.py --dataset "D:\path\to\images" --limit 1000
```

生成结果位于 `server/data/features.npz` 和 `server/data/image_paths.json`。这些文件可以重新生成，因此不会提交到 Git。

## 启动服务

```powershell
python server/app.py
```

浏览器访问 `http://127.0.0.1:5000`，上传一张不超过 10 MB 的 PNG 或 JPEG 图片即可检索。

## 上传 GitHub 前说明

本地可能仍保留旧版 `server.zip`、`saved_features_recom.txt`、数据集和 TensorFlow 模型。它们均已被 `.gitignore` 排除，不会通过正常的 `git add .` 上传。

提交前建议运行：

```powershell
git status --ignored
```

确认大型本地文件显示在 ignored 列表中。

