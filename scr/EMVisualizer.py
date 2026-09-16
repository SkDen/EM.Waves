
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
import glob
import cv2
from typing import Optional, Tuple, List, Union
import warnings


class EMVisualizer:
    """
    Визуализация результатов моделирования электромагнитных волн.
    Загружает серию снимков (двумерных массивов) и создаёт анимацию
    с возможностью сохранения в видеоформат.
    """

    def __init__(self, snapshot_dir: str, file_pattern: str = "snapshot_{:06d}.npy",
                 colormap: str = 'RdBu', vmin: Optional[float] = None,
                 vmax: Optional[float] = None, interpolation: str = 'bilinear'):
        """
        Параметры
        ----------
        snapshot_dir : str
            Директория с файлами снимков.
        file_pattern : str, optional
            Шаблон имени файла с номером шага (по умолчанию "snapshot_{:06d}.npy").
        colormap : str, optional
            Цветовая карта для imshow (по умолчанию 'RdBu').
        vmin, vmax : float, optional
            Минимальное и максимальное значения для цветовой шкалы.
            Если не указаны, будут определены автоматически по всем снимкам.
        interpolation : str, optional
            Метод интерполяции для imshow (по умолчанию 'bilinear').
        """
        self.snapshot_dir = snapshot_dir
        self.file_pattern = file_pattern
        self.colormap = colormap
        self.interpolation = interpolation

        # Получаем список всех файлов, соответствующих шаблону
        self.snapshot_files = self._get_snapshot_files()
        if not self.snapshot_files:
            raise FileNotFoundError(f"Не найдено файлов по шаблону {file_pattern} в {snapshot_dir}")

        # Загружаем первый снимок для определения размера
        first_snapshot = self.load_snapshot(0)
        self.shape = first_snapshot.shape
        self.dtype = first_snapshot.dtype

        # Если vmin/vmax не заданы, вычисляем глобальный диапазон
        if vmin is None or vmax is None:
            self.vmin, self.vmax = self._compute_global_range()
        else:
            self.vmin, self.vmax = vmin, vmax

        print(f"Найдено {len(self.snapshot_files)} снимков, размер {self.shape}, "
              f"диапазон значений: [{self.vmin:.3f}, {self.vmax:.3f}]")

    def _get_snapshot_files(self) -> List[str]:
        """Возвращает отсортированный список файлов снимков."""
        # Используем glob для поиска всех файлов, соответствующих шаблону с любым номером
        # Шаблон: заменим {:06d} на * (любая последовательность цифр)
        pattern = self.file_pattern.replace("{:06d}", "*")
        full_pattern = os.path.join(self.snapshot_dir, pattern)
        files = glob.glob(full_pattern)
        # Сортируем по номеру, извлекая число из имени
        def extract_number(fname):
            # Предполагаем, что номер — последняя группа цифр перед расширением
            import re
            numbers = re.findall(r'\d+', os.path.basename(fname))
            return int(numbers[-1]) if numbers else 0
        files.sort(key=extract_number)
        return files

    def load_snapshot(self, step: int) -> np.ndarray:
        """
        Загружает снимок по номеру шага (индекс в списке файлов).
        Если step выходит за пределы, выбрасывает исключение.
        """
        if step < 0 or step >= len(self.snapshot_files):
            raise IndexError(f"Шаг {step} вне диапазона (0..{len(self.snapshot_files)-1})")
        return np.load(self.snapshot_files[step])

    def _compute_global_range(self) -> Tuple[float, float]:
        """
        Вычисляет глобальный минимум и максимум по всем снимкам.
        Для больших данных можно использовать выборку, но здесь полный проход.
        """
        vmin = np.inf
        vmax = -np.inf
        for f in self.snapshot_files:
            data = np.load(f)
            vmin = min(vmin, data.min())
            vmax = max(vmax, data.max())
        return vmin, vmax

    def get_frame(self, step: int) -> np.ndarray:
        """
        Возвращает кадр в виде массива float (оригинальные данные).
        """
        return self.load_snapshot(step)

    def show_frame(self, step: int, ax=None, **imshow_kwargs):
        """
        Отображает один кадр в matplotlib окне.
        Если ax не задан, создаёт новую фигуру.
        Дополнительные аргументы передаются в imshow.
        """
        data = self.load_snapshot(step)
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        im = ax.imshow(data, cmap=self.colormap, vmin=self.vmin, vmax=self.vmax,
                       interpolation=self.interpolation, **imshow_kwargs)
        ax.set_title(f"Шаг {step}")
        plt.colorbar(im, ax=ax)
        plt.show()

    def animate_interactive(self, interval: int = 100, repeat: bool = True):
        """
        Интерактивная анимация в окне matplotlib.
        Параметры:
        interval : int
            Задержка между кадрами в миллисекундах.
        repeat : bool
            Повторять ли анимацию по кругу.
        """
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        first_data = self.load_snapshot(0)
        im = ax.imshow(first_data, cmap=self.colormap, vmin=self.vmin, vmax=self.vmax,
                       interpolation=self.interpolation)
        plt.colorbar(im, ax=ax)
        ax.set_title("Шаг 0")

        def update(frame):
            data = self.load_snapshot(frame)
            im.set_array(data)
            ax.set_title(f"Шаг {frame}")
            return [im]

        ani = animation.FuncAnimation(fig, update, frames=len(self.snapshot_files),
                                      interval=interval, blit=True, repeat=repeat)
        plt.show()
        return ani

    def create_video(self, output_file: str, fps: int = 10, dpi: int = 100,
                     figsize: Tuple[int, int] = (8, 6), use_opencv: bool = False,
                     codec: str = 'mp4v', bitrate: int = 1800):
        """
        Создаёт видео из последовательности снимков.

        Параметры
        ----------
        output_file : str
            Имя выходного видеофайла (например, 'animation.mp4').
        fps : int
            Кадров в секунду.
        dpi : int
            Разрешение для matplotlib (если use_opencv=False).
        figsize : tuple
            Размер фигуры в дюймах (ширина, высота).
        use_opencv : bool
            Если True, использует OpenCV для быстрой записи (без matplotlib).
            Если False, использует matplotlib.animation (медленнее, но более настраиваемо).
        codec : str
            Кодек OpenCV (по умолчанию 'mp4v').
        bitrate : int
            Битрейт для matplotlib writer.
        """
        if use_opencv:
            self._create_video_opencv(output_file, fps, codec)
        else:
            self._create_video_matplotlib(output_file, fps, dpi, figsize, bitrate)

    def _create_video_matplotlib(self, output_file: str, fps: int, dpi: int,
                                  figsize: Tuple[int, int], bitrate: int):
        """
        Создание видео с помощью matplotlib.animation.
        """
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        first_data = self.load_snapshot(0)
        im = ax.imshow(first_data, cmap=self.colormap, vmin=self.vmin, vmax=self.vmax,
                       interpolation=self.interpolation)
        plt.colorbar(im, ax=ax)
        ax.set_title("Шаг 0")

        def update(frame):
            data = self.load_snapshot(frame)
            im.set_array(data)
            ax.set_title(f"Шаг {frame}")
            return [im]

        ani = animation.FuncAnimation(fig, update, frames=len(self.snapshot_files),
                                      interval=1000/fps, blit=True)

        # Сохранение видео
        writer = animation.FFMpegWriter(fps=fps, bitrate=bitrate)
        ani.save(output_file, writer=writer, dpi=dpi)
        plt.close(fig)
        print(f"Видео сохранено: {output_file}")

    def _create_video_opencv(self, output_file: str, fps: int, codec: str):
        """
        Создание видео с помощью OpenCV (быстрее, не требует matplotlib).
        Предполагается, что данные нормируются в диапазон [0,255] и конвертируются в uint8.
        """
        # Определяем размер кадра по первому снимку
        first_data = self.load_snapshot(0)
        height, width = first_data.shape

        # Инициализируем VideoWriter
        fourcc = cv2.VideoWriter_fourcc(*codec)
        out = cv2.VideoWriter(output_file, fourcc, fps, (width, height), isColor=True)

        for idx in range(len(self.snapshot_files)):
            data = self.load_snapshot(idx)
            # Нормализация в диапазон [0,255] с учётом vmin, vmax
            # Если vmin == vmax, избегаем деления на ноль
            if self.vmax - self.vmin < 1e-12:
                norm_data = np.zeros_like(data)
            else:
                norm_data = (data - self.vmin) / (self.vmax - self.vmin)
                norm_data = np.clip(norm_data, 0, 1)
            # Преобразуем в 8-битное изображение
            img_uint8 = (norm_data * 255).astype(np.uint8)
            # Применяем цветовую карту (OpenCV требует 3 канала)
            # Используем colormap matplotlib для конвертации
            cmap = plt.get_cmap(self.colormap)
            colored = cmap(norm_data.flatten())[:, :3]  # берём RGB, отбрасываем alpha
            colored = (colored * 255).astype(np.uint8).reshape(height, width, 3)
            # OpenCV использует BGR, конвертируем
            colored_bgr = cv2.cvtColor(colored, cv2.COLOR_RGB2BGR)
            out.write(colored_bgr)

        out.release()
        print(f"Видео сохранено: {output_file}")

    def show_data_range(self):
        """Выводит информацию о диапазоне значений."""
        print(f"Диапазон значений по всем снимкам: [{self.vmin:.3f}, {self.vmax:.3f}]")

    def __len__(self):
        return len(self.snapshot_files)