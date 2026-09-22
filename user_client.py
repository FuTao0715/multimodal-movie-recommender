# user_client.py  用户版客户端
import time
import socket
import webview
import ctypes

SERVICE_PORT = 5000
SERVICE_URL = "http://127.0.0.1:5000/user"
MB_OK = 0x0
MB_ICONERROR = 0x10

def is_port_in_use(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(("127.0.0.1", port))
            return False
        except OSError:
            return True

# 等待后台服务启动
wait_time = 0
while wait_time < 60:
    if is_port_in_use(SERVICE_PORT):
        break
    time.sleep(1)
    wait_time += 1

if not is_port_in_use(SERVICE_PORT):
    ctypes.windll.user32.MessageBoxW(
        0,
        "未检测到系统服务，请先启动服务程序后再打开客户端！",
        "启动失败",
        MB_OK | MB_ICONERROR
    )
    exit()

# 打开用户版窗口
window = webview.create_window(
    title="电影推荐系统",
    url=SERVICE_URL,
    width=1280,
    height=800,
    resizable=True,
    min_size=(960, 600)
)
webview.start()