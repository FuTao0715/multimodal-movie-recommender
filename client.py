# client.py 只负责弹出窗口，完全不导入torch
import time
import socket
import webview
import ctypes

SERVICE_PORT = 5000
SERVICE_URL = "http://127.0.0.1:5000"
MB_OK = 0x0
MB_ICONERROR = 0x10

# 检测端口是否启动（判断服务是否运行）
def is_port_in_use(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(("127.0.0.1", port))
            return False
        except OSError:
            return True

# 等待服务启动，最多等待60秒
wait_time = 0
while wait_time < 60:
    if is_port_in_use(SERVICE_PORT):
        break
    time.sleep(1)
    wait_time += 1

# 判断超时未启动，弹窗提示
if not is_port_in_use(SERVICE_PORT):
    ctypes.windll.user32.MessageBoxW(
        0,
        "未检测到后台服务，请先双击运行 start_server.py 启动模型服务后再打开客户端！",
        "启动失败",
        MB_OK | MB_ICONERROR
    )
    exit()

# 服务正常，弹出内嵌网页窗口
window = webview.create_window(
    title="多模态电影推荐系统 客户端",
    url=SERVICE_URL,
    width=1400,
    height=900,
    resizable=True,
    min_size=(1000, 600)
)
webview.start()