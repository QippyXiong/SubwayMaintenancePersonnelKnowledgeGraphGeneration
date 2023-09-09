from random import randint
from threading import Thread
from time import sleep

from animator import Animator

if __name__ == '__main__':
    # test animator func
    a = Animator('fucyou man', 'thankyou', lengends=['h1', 'f1'], x_lim=[0, 10], y_lim=[0, 10])

    def test_anima(ani : Animator) -> None:
        for i in range(1, 10):
            sleep(0.5)
            ani.add(i, [randint(2, 10), randint(1,8)])

    addThrd = Thread(target=lambda: test_anima(a))

    addThrd.start()
    a.show()
    addThrd.join()
