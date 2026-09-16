# main.py

import numpy as np
import os
import sys
import time


from config         import SimulationConfig, BoundaryType
from EMPWave        import Wave2D_GPU
from EMVisualizer   import EMVisualizer
from kernel_loader  import OpenCLKernelLoader


def gaussian_source(x, y, t, x0=0.0, y0=0.0, sigma=0.2, freq=30.0):
    """
    Пример источника: осциллирующий гауссов пакет.
    f(x,y,t) = A * exp(-((x-x0)^2+(y-y0)^2)/(2*sigma^2)) * sin(2*pi*freq*t)
    """
    
    r2 = (x - x0)**2 + (y - y0)**2
    spatial = np.exp(-r2 / (2 * sigma**2))
    temporal = np.sin(2 * np.pi * freq * t)
    return spatial * temporal


def main():
    # Параметры моделирования
    nx = 1000
    ny = 1000
    dx = 0.01
    dy = 0.01
    dt = 0.005
    d:int = 2
    c = 0.3        # скорость волны
    c_wl = c        # скорость волны в фиктивнос слое
    steps = 50_000
    snap_interval = 100
    output_dir = './results'
    kernel_file = 'scr/computing_cores/wave2d.cl'
    platform = 0
    device = 0
    video = "./out_video/test.mp4"
    no_cfl_check = False      # Исправлено: булево значение
    dtype = np.float32        # Исправлено: float32 для совместимости с ядром

    # Пример различных сред (массив скоростей для узлов)
    c_mass = np.full((nx+2*d, ny+2*d), c, dtype=dtype)
    # c_mass[:, (ny+2*d)//2 + 200:] *= 2

    # Граничные условия для различных сторон сетки
    bc = {  
        'left':     BoundaryType.ABSORBING,
        'right':    BoundaryType.ABSORBING,
        'bottom':   BoundaryType.ABSORBING,
        'top':      BoundaryType.ABSORBING
        }

    # Проверка существования файла с ядрами
    if not os.path.exists(kernel_file):
        print(f"Ошибка: файл ядра {kernel_file} не найден.")
        sys.exit(1)

    # Создание конфигурации
    config = SimulationConfig(
        nx=nx,
        ny=ny,
        dx=dx,
        dy=dy,
        dt=dt,
        d=d,
        c_wl=c_wl,
        c=c_mass,
        bc=bc,
        use_source=False,
        source=lambda x, y, t: gaussian_source(x, y, t, x0=nx*dx/2, y0=ny*dy/2),
        snapshot_interval=snap_interval,
        output_dir=output_dir,
        dtype=dtype
    )

    # Проверка условия Куранта
    if not no_cfl_check:
        if not config.check_cfl():
            print("Предупреждение: условие Куранта не выполнено! Возможна неустойчивость.")
        else:
            print("Условие Куранта выполнено.")

    # Создание экземпляра модели
    model = Wave2D_GPU(config)

    # # Начальное условие: гауссов импульс в центре
    # def init_gaussian(x, y):
    #     x0 = config.nx * config.dx / 2
    #     y0 = config.ny * config.dy / 2
    #     sigma = 0.2
    #     return np.exp(-((x - x0)**2 + (y - y0)**2) / (2 * sigma**2))

    # model.set_initial_condition(init_gaussian)

    # Инициализация OpenCL
    model.setup_opencl(platform_idx=platform, device_idx=device, kernel_path=kernel_file)

    print("Запуск моделирования...")
    start = time.time()
    # try:
    model.run(num_steps=steps, snapshot_interval=snap_interval, output_dir=output_dir)
    # except Exception as e:
    #     print(f"Ошибка во время моделирования: {e}")
    #     model.free_resources()
    #     sys.exit(1)
    end = time.time()
    print(f"Моделирование завершено за {end - start:.2f} с.")

    # Визуализация, если указан файл видео
    if video:
        print("Создание видео...")
        try:
            amp = 0.05
            viz = EMVisualizer(output_dir, vmin=-amp, vmax=amp)
            viz.create_video(video, fps=20, use_opencv=True)
        except Exception as e:
            print(f"Ошибка при создании видео (OpenCV): {e}")
            try:
                viz.create_video(video, fps=20, use_opencv=False)
            except Exception as e2:
                print(f"Не удалось создать видео (matplotlib): {e2}")
    else:
        print("Для создания видео укажите --video имя_файла.mp4")


if __name__ == "__main__":
    main()