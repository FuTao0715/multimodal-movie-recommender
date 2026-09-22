#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
import os
import time
import threading
import socket
import webview
import ctypes

# -------------------------- 【关键修正】路径适配 --------------------------
def get_base_path():
    if hasattr(sys, '_MEIPASS'):
        # 打包后：所有资源文件都在 PyInstaller 的运行根目录 _MEIPASS 中
        return sys._MEIPASS
    # 本地开发环境：项目根目录
    return r"D:\VS_CODE\机器学习"

BASE_DIR = get_base_path()
# 强制切换工作目录，确保 app.py 能找到所有资源
os.chdir(BASE_DIR)

SERVICE_PORT = 5000
SERVICE_URL = f"http://127.0.0.1:{SERVICE_PORT}"
MB_OK = 0x0
MB_ICONERROR = 0x10

# -------------------------- 端口检测 --------------------------
def is_port_in_use(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(("127.0.0.1", port))
            return False
        except OSError:
            return True

# -------------------------- 后台启动 Flask 服务（带异常捕获） --------------------------
service_error = None

def run_flask_service():
    global service_error
    try:
        from app import app
        app.run(
            host="127.0.0.1",
            port=SERVICE_PORT,
            debug=False,
            use_reloader=False,
            threaded=True
        )
    except Exception as e:
        service_error = str(e)
        import traceback
        traceback.print_exc()

# -------------------------- 主入口 --------------------------
if __name__ == "__main__":
    # 1. 单例检测：已经运行就直接打开窗口
    if is_port_in_use(SERVICE_PORT):
        webview.create_window(
            title="多模态电影推荐系统 客户端",
            url=SERVICE_URL,
            width=1400,
            height=900,
            resizable=True
        )
        webview.start()
        sys.exit(0)

    # 2. 后台启动服务
    threading.Thread(target=run_flask_service, daemon=True).start()

    # 3. 等待服务启动（最多等 40 秒，适配模型加载慢）
    waited = 0
    service_ready = False
    while waited < 40:
        if is_port_in_use(SERVICE_PORT):
            service_ready = True
            break
        if service_error is not None:
            # 服务启动报错，弹窗提示并退出
            ctypes.windll.user32.MessageBoxW(
                0,
                f"服务启动失败：{service_error}",
                "错误",
                MB_OK | MB_ICONERROR
            )
            sys.exit(1)
        time.sleep(1)
        waited += 1

    # 4. 服务就绪，打开客户端窗口
    if service_ready:
        window = webview.create_window(
            title="多模态电影推荐系统 客户端",
            url=SERVICE_URL,
            width=1400,
            height=900,
            resizable=True,
            min_size=(1000, 600)
        )
        webview.start()
    else:
        ctypes.windll.user32.MessageBoxW(
            0,
            "服务启动超时，请检查是否有其他程序占用5000端口",
            "启动失败",
            MB_OK | MB_ICONERROR
        )

    sys.exit(0)