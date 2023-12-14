import PySimpleGUI as sg
import subprocess  # 用于运行外部命令

# 定义 GUI 布局
layout = [
    [sg.Text("Hello, PySimpleGUI!")],
    [sg.Button("OK")]
]

# 创建窗口
window = sg.Window("My First GUI", layout)

# 事件循环
while True:
    event, values = window.read()

    # 处理事件
    if event == "OK":
        # 在这里添加运行 test_model.py 的代码
        try:
            subprocess.run(["python", "test_model.py"])  # 运行 test_model.py
        except Exception as e:
            sg.popup_error(f"Error running a.py: {e}")
    if event == sg.WINDOW_CLOSED:
        break
# 关闭窗口
window.close()
