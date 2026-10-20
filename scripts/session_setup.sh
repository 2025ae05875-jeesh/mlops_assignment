#!/usr/bin/env bash
# Run at the start of EVERY lab session:   source scripts/session_setup.sh
# The lab VM changes between sessions; only /home/cloud/workspace persists.
PROJECT=/home/cloud/workspace/mlops_assignment
VENV="$HOME/.venvs/heartml"          # local disk: fast, recreated if missing
export PATH="/home/cloud/workspace/bin:$PATH"   # persistent kubectl location
cd "$PROJECT" || return 1
if ! "$VENV/bin/python" -c "import heartml, sklearn, mlflow, fastapi" 2>/dev/null; then
  echo "[setup] environment missing or incomplete -> installing (3-6 min)"
  [ -x "$VENV/bin/python" ] || python3 -m venv "$VENV"
  "$VENV/bin/pip" install --progress-bar off -q --upgrade pip || return 1
  "$VENV/bin/pip" install --progress-bar off -r requirements.txt \
    || { echo "[setup] INSTALL FAILED - copy the error above"; return 1; }
  "$VENV/bin/pip" install --progress-bar off -q -e . || return 1
fi
source "$VENV/bin/activate"
echo "[setup] python: $(which python) | $(python --version)"
python -c "import heartml, sklearn, mlflow, fastapi; print('[setup] heartml', heartml.__version__, '| sklearn', sklearn.__version__, '| mlflow', mlflow.__version__, '| fastapi', fastapi.__version__)"
