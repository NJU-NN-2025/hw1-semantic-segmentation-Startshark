@echo off
REM 一键运行脚本 - Windows PowerShell
REM LLM 辅助: 本文件由 GitHub Copilot 辅助生成

echo ============================================================
echo 南极动物语义分割 - 一键运行脚本
echo ============================================================
echo.

REM 检查 Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [错误] 未检测到 Python，请先安装 Python 3.8+
    pause
    exit /b 1
)

echo [1/7] 环境验证测试...
python scripts/quick_test.py
if errorlevel 1 (
    echo.
    echo [错误] 环境验证失败，请先配置环境:
    echo   pip install -r requirements.txt
    pause
    exit /b 1
)

echo.
echo [2/7] 检查数据集...
if not exist "dataset" (
    echo [警告] 未找到 dataset 目录
    echo 请下载数据集并放入 dataset/ 目录
    echo 数据集链接: https://box.nju.edu.cn/d/74c94657a0404eb79c74/
    pause
    exit /b 1
)

echo.
echo [3/7] 数据预处理...
set /p preprocess="是否运行数据预处理? (y/n): "
if /i "%preprocess%"=="y" (
    python scripts/preprocess_data.py
    if errorlevel 1 (
        echo [错误] 数据预处理失败
        pause
        exit /b 1
    )
)

echo.
echo [4/7] 模型训练...
set /p train="是否开始训练? (y/n): "
if /i "%train%"=="y" (
    set /p model_size="选择模型大小 (n/s/m, 默认n): "
    if "%model_size%"=="" set model_size=n
    
    set /p epochs="训练轮数 (默认100): "
    if "%epochs%"=="" set epochs=100
    
    echo 开始训练 YOLOv8%model_size%-seg，共 %epochs% 轮...
    python scripts/train.py --model %model_size% --epochs %epochs%
    if errorlevel 1 (
        echo [错误] 训练失败
        pause
        exit /b 1
    )
)

echo.
echo [5/7] 模型测试...
set /p test="是否测试模型? (y/n): "
if /i "%test%"=="y" (
    if not exist "runs\segment\antarctic_yolo\weights\best.pt" (
        echo [错误] 未找到训练好的模型
        pause
        exit /b 1
    )
    
    python scripts/test.py --model runs\segment\antarctic_yolo\weights\best.pt --source test
    if errorlevel 1 (
        echo [错误] 测试失败
        pause
        exit /b 1
    )
)

echo.
echo [6/7] 模型评估...
set /p eval="是否评估模型? (y/n): "
if /i "%eval%"=="y" (
    python scripts/evaluate.py --model runs\segment\antarctic_yolo\weights\best.pt --val-data data\processed\val\images
    if errorlevel 1 (
        echo [错误] 评估失败
        pause
        exit /b 1
    )
)

echo.
echo [7/7] 完成!
echo ============================================================
echo 工作流程已完成!
echo 结果位置:
echo   - 训练输出: runs\segment\antarctic_yolo\
echo   - 测试结果: test_results\predictions\
echo   - 评估结果: evaluation_results.json
echo ============================================================
echo.
pause
