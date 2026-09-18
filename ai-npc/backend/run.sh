#!/usr/bin/env bash
# =============================================================================
# 启动脚本（run.sh）
# =============================================================================
# 用法：在 backend 目录下执行 ./run.sh
# 作用：使用「项目根目录的 venv」启动 FastAPI 后端，并自动加载项目根目录的 .env
# 这样不需要在 backend 下再建一个 venv，和 notebook 等共用一套依赖即可。
# =============================================================================

# 进入脚本所在目录（即 backend），保证后续 pwd 是 backend
cd "$(dirname "$0")"

# 上两级目录 = 仓库根目录（含 venv / .env）
ROOT="$(cd ../.. && pwd)"

# 若项目根下存在 venv，就激活它；否则提示用户先创建并安装依赖
if [ -d "$ROOT/venv" ]; then
  source "$ROOT/venv/bin/activate"
else
  echo "请先在项目根目录创建 venv 并安装依赖: python -m venv venv && source venv/bin/activate && pip install -r ai-npc/backend/requirements.txt"
  exit 1
fi

# 把当前目录（backend）设为 Python 的模块搜索路径，这样 import config、main 等才能找到
export PYTHONPATH="$(pwd)"

# 启动 uvicorn：main:app 表示 main.py 里的变量 app；--reload 表示改代码后自动重启
uvicorn main:app --reload --host 0.0.0.0 --port 8000
