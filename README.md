<div style="text-align: center;"><h1>ExterKB</h1></div>

## 简述

`ExterKB` 是一个基于 `LlamaIndex` + `Qdrant` + `FastApi` 的 **LLM 外部知识库实现**。

## 当前开发方向

1. 实现完整的 RAG 知识库管理功能
2. 实现两种检索方式
3. 向量检索端点面向 **对接 Dify** 的外部知识库 API开发

## 目标

1. [ ] 实现 RAG 知识库的增删改管理功能
    1. [ ] 从文件导入
    2. [ ] 从文本新建
2. [ ] 实现高度自定义化的元数据标记
3. [ ] 实现两种基础的查询方式
    1. [ ] 基于 Qdrant 元数据过滤的精确检索(~~当成关系型库用~~)
    2. [ ] 基于向量的语义化检索

## 项目结构

```
.
├── api
│    └── endpoints
├── service
├── util
├── config.py
└── main.py
```

一目了然，不多解释。

## 开发调试

克隆仓库，安装依赖，部署 Qdrant 服务端，按需配置 `config.py`，运行 `main.py`。

项目前后端分离，此仓库仅实现后端。