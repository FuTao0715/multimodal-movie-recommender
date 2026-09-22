# start_server.py 仅本地运行，不要打包这个文件
import sys
import os
import time

# 统一路径逻辑，和app.py完全一致
def get_base_path():
    return os.path.dirname(os.path.abspath(__file__))

BASE_DIR = get_base_path()
os.chdir(BASE_DIR)

SERVICE_PORT = 5000

if __name__ == "__main__":
    # 直接启动Flask服务（包含torch、模型加载）
    from app import app
    print("服务正在启动，加载模型中，请等待...")
    # 关键修改 host=127.0.0.1
    app.run(
        host="127.0.0.1",
        port=SERVICE_PORT,
        debug=False,
        use_reloader=False
    )