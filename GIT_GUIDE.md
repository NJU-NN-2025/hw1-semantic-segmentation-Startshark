# Git 提交参考

## 初始提交

```bash
git add .
git commit -m "feat: 初始化南极动物语义分割项目

- 添加项目依赖配置 (requirements.txt, environment.yml)
- 实现数据预处理脚本 (SAM 伪标签生成)
- 实现 YOLOv8-seg 训练脚本 (适配 RTX 4060)
- 实现模型测试和评估脚本 (7个评估指标)
- 添加快速验证测试
- 添加完整项目文档

LLM 辅助: GitHub Copilot 辅助生成所有脚本和文档"
```

## 后续提交示例

### 添加数据集
```bash
git add dataset/
git commit -m "data: 添加训练数据集

- 下载南极动物数据集
- 数据集包含 XXX 张图像
- 遵守数据集使用协议，仅用于课程作业"
```

### 添加训练结果
```bash
git add runs/
git commit -m "train: 完成模型训练

- 训练 YOLOv8n-seg 模型
- Epochs: 100
- Best mIoU: XX.XX
- 训练时长: X 小时"
```

### 添加测试结果
```bash
git add test/
git add test_results/
git commit -m "test: 添加测试集结果

- 对测试集进行预测
- 生成可视化结果
- 保存至 test/ 文件夹供论文使用"
```

### 添加评估结果
```bash
git add evaluation_results.json
git commit -m "eval: 添加模型评估结果

- mIoU: XX.XX
- Dice: XX.XX
- Pixel Accuracy: XX.XX
- 详见 evaluation_results.json"
```

## 提交最佳实践

### 1. 清晰的提交信息
- 使用 feat/fix/docs/test 等前缀
- 简洁描述改动内容
- 必要时添加详细说明

### 2. 合理的提交粒度
- 一次提交完成一个功能
- 避免混合多个不相关的改动
- 便于回滚和查找问题

### 3. 遵守 .gitignore
- 不提交模型权重 (.pt 文件太大)
- 不提交原始数据集（版权问题）
- 不提交临时文件和缓存

### 4. 标注 LLM 使用
- 在提交信息中说明 LLM 辅助
- 保持透明和诚信

## Git 工作流示例

```bash
# 1. 初始化仓库
git init
git add .
git commit -m "feat: 初始化项目"

# 2. 添加远程仓库
git remote add origin <your-repo-url>

# 3. 推送到 GitHub
git push -u origin main

# 4. 后续更新
git add <files>
git commit -m "type: message"
git push

# 5. 创建标签（重要版本）
git tag -a v1.0 -m "完成基础功能"
git push --tags
```

## .gitignore 说明

已配置忽略:
- Python 缓存 (__pycache__, *.pyc)
- 虚拟环境 (venv/, env/)
- 数据集 (dataset/, data/, *.jpg, *.png)
- 模型权重 (*.pt, *.pth, *.onnx)
- 训练输出 (runs/, logs/, weights/)
- IDE 配置 (.vscode/, .idea/)

不忽略:
- test/ 目录中的结果（用于论文）
- requirements.txt, environment.yml
- README.md 和其他文档
- 所有 Python 脚本

## 注意事项

### ⚠️ 不要提交的文件
- ❌ 原始数据集图像（版权限制）
- ❌ 大型模型权重文件（使用 Git LFS 或云存储）
- ❌ 临时文件和缓存
- ❌ 包含敏感信息的配置

### ✅ 应该提交的文件
- ✅ 所有 Python 脚本
- ✅ 配置文件 (requirements.txt, .yaml)
- ✅ 文档 (README.md, *.md)
- ✅ 测试结果可视化（少量图像）
- ✅ 评估结果 JSON

## GitHub Classroom 提交

对于 GitHub Classroom 作业:

```bash
# 1. 接受作业邀请（创建个人仓库）
# 2. 克隆到本地
git clone <classroom-repo-url>
cd <repo-name>

# 3. 添加所有文件
git add .
git commit -m "feat: 完成语义分割作业

- 实现完整的训练和测试流程
- 评估指标 ≥5 个
- 包含测试集结果

LLM 辅助: 使用 GitHub Copilot 辅助开发"

# 4. 推送到 GitHub
git push origin main

# 5. 验证提交
# 访问 GitHub 仓库确认所有文件已上传
```

---

**最后检查清单**:
- [ ] 所有脚本可以正常运行
- [ ] README.md 内容完整
- [ ] 测试结果已保存在 test/ 目录
- [ ] LLM 使用已在代码和文档中标注
- [ ] 遵守数据集使用协议（未上传原始数据）
- [ ] .gitignore 配置正确
