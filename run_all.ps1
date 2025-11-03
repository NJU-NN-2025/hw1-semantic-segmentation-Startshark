# 一键运行脚本 - PowerShell
# LLM 辅助: 本文件由 GitHub Copilot 辅助生成

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "南极动物语义分割 - 一键运行脚本" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

# 检查 Python
try {
    $pythonVersion = python --version 2>&1
    Write-Host "✓ 检测到 Python: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "✗ 未检测到 Python，请先安装 Python 3.8+" -ForegroundColor Red
    exit 1
}

# 步骤1: 环境验证
Write-Host "`n[1/7] 环境验证测试..." -ForegroundColor Yellow
python scripts/quick_test.py
if ($LASTEXITCODE -ne 0) {
    Write-Host "`n✗ 环境验证失败，请先配置环境:" -ForegroundColor Red
    Write-Host "  pip install -r requirements.txt" -ForegroundColor Yellow
    exit 1
}

# 步骤2: 检查数据集
Write-Host "`n[2/7] 检查数据集..." -ForegroundColor Yellow
if (-not (Test-Path "dataset")) {
    Write-Host "⚠ 未找到 dataset 目录" -ForegroundColor Red
    Write-Host "请下载数据集并放入 dataset/ 目录" -ForegroundColor Yellow
    Write-Host "数据集链接: https://box.nju.edu.cn/d/74c94657a0404eb79c74/" -ForegroundColor Cyan
    exit 1
} else {
    Write-Host "✓ 找到 dataset 目录" -ForegroundColor Green
}

# 步骤3: 数据预处理
Write-Host "`n[3/7] 数据预处理..." -ForegroundColor Yellow
$preprocess = Read-Host "是否运行数据预处理? (y/n)"
if ($preprocess -eq 'y') {
    python scripts/preprocess_data.py
    if ($LASTEXITCODE -ne 0) {
        Write-Host "✗ 数据预处理失败" -ForegroundColor Red
        exit 1
    }
}

# 步骤4: 模型训练
Write-Host "`n[4/7] 模型训练..." -ForegroundColor Yellow
$train = Read-Host "是否开始训练? (y/n)"
if ($train -eq 'y') {
    $modelSize = Read-Host "选择模型大小 (n/s/m, 默认n)"
    if ([string]::IsNullOrEmpty($modelSize)) { $modelSize = "n" }
    
    $epochs = Read-Host "训练轮数 (默认100)"
    if ([string]::IsNullOrEmpty($epochs)) { $epochs = "100" }
    
    Write-Host "开始训练 YOLOv8$modelSize-seg，共 $epochs 轮..." -ForegroundColor Cyan
    python scripts/train.py --model $modelSize --epochs $epochs
    if ($LASTEXITCODE -ne 0) {
        Write-Host "✗ 训练失败" -ForegroundColor Red
        exit 1
    }
}

# 步骤5: 模型测试
Write-Host "`n[5/7] 模型测试..." -ForegroundColor Yellow
$test = Read-Host "是否测试模型? (y/n)"
if ($test -eq 'y') {
    $modelPath = "runs/segment/antarctic_yolo/weights/best.pt"
    if (-not (Test-Path $modelPath)) {
        Write-Host "✗ 未找到训练好的模型: $modelPath" -ForegroundColor Red
        exit 1
    }
    
    python scripts/test.py --model $modelPath --source test
    if ($LASTEXITCODE -ne 0) {
        Write-Host "✗ 测试失败" -ForegroundColor Red
        exit 1
    }
}

# 步骤6: 模型评估
Write-Host "`n[6/7] 模型评估..." -ForegroundColor Yellow
$eval = Read-Host "是否评估模型? (y/n)"
if ($eval -eq 'y') {
    $modelPath = "runs/segment/antarctic_yolo/weights/best.pt"
    python scripts/evaluate.py --model $modelPath --val-data data/processed/val/images
    if ($LASTEXITCODE -ne 0) {
        Write-Host "✗ 评估失败" -ForegroundColor Red
        exit 1
    }
}

# 完成
Write-Host "`n[7/7] 完成!" -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "工作流程已完成!" -ForegroundColor Green
Write-Host "结果位置:" -ForegroundColor Cyan
Write-Host "  - 训练输出: runs/segment/antarctic_yolo/" -ForegroundColor Yellow
Write-Host "  - 测试结果: test_results/predictions/" -ForegroundColor Yellow
Write-Host "  - 评估结果: evaluation_results.json" -ForegroundColor Yellow
Write-Host "============================================================" -ForegroundColor Cyan
