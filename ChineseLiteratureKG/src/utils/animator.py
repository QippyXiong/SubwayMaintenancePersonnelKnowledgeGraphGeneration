# this animator class using matplot lib for showing the trainning process
# we can also use tqdm

import tqdm
from matplotlib import figure, pyplot as plt
from threading import Thread
from typing import *

class Animator():
    r"""
        这是一个使用matplotlib显示训练过程的动画类，类似于李沐课程中的，不过李沐使用的是jupyter inline显示图表，因而实现起来有所不同
        你需要多线程手段，才能保证显示无问题
        使用样例：
        model_thread = Thread(lambda: train(model, animator, epoch, ...))
        model_thread.start()
        animator.show() # 阻塞主线程
        model_thread.join()
        其中：
        def train(...):
            ...
            animator.add(epoch_num, x, y)
    """

    def __init__(self, 
                    x_label: Union[str, None] = None,
                    y_label: Union[str, None] = None,
                    lengends: Union[list[str], None] = None,
                    x_lim : Union[tuple[float, float], None] = None, 
                    y_lim : Union[tuple[float, float], None] = None,
                    x_scale : str = 'linear',
                    y_scale : str = 'linear',
                    fmt : tuple[str] = ('-', 'm--', 'g-.', 'r:'),
                    size : tuple[float] = (4, 3)
                ) -> None:
        r"""
            labels是每个y对应的名称，x_lim，y_lim是整个图表的最大最小值范围
            x_scale, y_scale是显示形式，默认为线性级（'linear'），可以选用对数级（'log'）
            fmts是显示线条格式，具体参数可查matplotlib，我觉得用默认的就好
        """
        self.fig, self.axe = plt.subplots(1, 1)
        self.fig.set_size_inches(size[0], size[1])
        self.y_label = y_label
        self.x_label = x_label
        self.legends = lengends
        self.x_lim = x_lim
        self.y_lim = y_lim
        self.x_scale = x_scale
        self.y_scale = y_scale
        self.fmt = fmt
        self.X = []
        self.Y = []
        self.clear()
        self.config()


    def config(self):

        if(self.x_lim): self.axe.set_xlim(self.x_lim)
        if(self.y_lim): self.axe.set_ylim(self.y_lim)

        if(self.x_label):
            self.axe.set_xlabel (self.x_label)
        if(self.y_label):
            self.axe.set_ylabel (self.y_label)
        self.axe.set_xscale (self.x_scale)
        self.axe.set_yscale (self.y_scale)
        if(self.legends):
             self.axe.legend(self.legends)
        self.axe.grid()

    
    def add(self, x: float, y: Union[ list[float], float]) -> None:
        r""" 在原先的图上添加一个点 """
        self.X.append(x)
        self.Y.append(y)
        self._print()

    
    def show(self, block : bool = True) -> None:
        r""" 不可在子线程调用此方法 """
        self.config()
        plt.subplots_adjust(left=0.15, bottom=0.2)
        plt.show(block = block)

    def plot(self, x : list[float], y : Union[list[list[float]], list[float]] ) -> None:
        r""" 打印x, y """
        self.X = x
        self.Y = y
        self._print()


    def clear(self) -> None:
        r""" 清空图表 """
        self.axe.cla()
        self.X = []
        self.Y = []
    

    def _print(self) -> None:
        self.axe.cla()
        self.axe.plot(self.X, self.Y, *self.fmt)
        self.config()
        self.fig.canvas.draw_idle()


    def flush_events(self) -> None:
        r""" 刷新接收的UI事件，非阻塞时使用 """
        self.fig.canvas.flush_events()
    
    
