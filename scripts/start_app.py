import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FRONTEND_DIR = ROOT / "frontend"
DIST_DIR = FRONTEND_DIR / "dist"


def ensure_frontend_build() -> None:
    if DIST_DIR.exists():
        return
    if not FRONTEND_DIR.exists():
        raise RuntimeError("frontend 目录不存在，请先生成前端工程。")
    print("Frontend dist 未找到，开始构建...")
    subprocess.check_call(["npm", "install"], cwd=str(FRONTEND_DIR))
    subprocess.check_call(["npm", "run", "build"], cwd=str(FRONTEND_DIR))


def run_react_api() -> None:
    os.chdir(ROOT)
    sys.path.insert(0, str(ROOT))
    ensure_frontend_build()
    import uvicorn

    uvicorn.run("scripts.web_api:app", host="0.0.0.0", port=8000, reload=False)


def run_comfyui() -> None:
    os.chdir(ROOT)
    sys.path.insert(0, str(ROOT))
    import scripts.comfyui_app as app

    app.build_ui().queue().launch(server_name="0.0.0.0", server_port=7860, theme=app.THEME, css=app.CSS)


if __name__ == "__main__":
    choice = ""
    while choice not in {"1", "2"}:
        print("Choose mode: 1 = react, 2 = comfyui")
        try:
            choice = input("Enter 1 or 2: ").strip()
        except EOFError:
            choice = "1"
    if choice == "2":
        run_comfyui()
    else:
        run_react_api()
