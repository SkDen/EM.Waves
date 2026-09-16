
import pyopencl as cl
import numpy as np
import os
from typing import Dict, Optional, Tuple, List


class OpenCLKernelLoader:
    """
    Загрузчик и компилятор OpenCL ядер.
    Предоставляет удобный интерфейс для выбора платформы и устройства,
    компиляции кода из файла и доступа к скомпилированным программам.
    """

    def __init__(self, platform_idx: int = 0, device_idx: int = 0):
        """
        Инициализация загрузчика: выбор платформы и устройства, создание контекста и очереди.

        Параметры
        ----------
        platform_idx : int, optional
            Индекс используемой платформы OpenCL (по умолчанию 0).
        device_idx : int, optional
            Индекс используемого устройства на выбранной платформе (по умолчанию 0).

        Исключения
        -----------
        IndexError
            Если указаны неверные индексы платформы или устройства.
        """
        # Получаем список доступных платформ
        self.platforms = cl.get_platforms()
        if platform_idx >= len(self.platforms):
            raise IndexError(f"Платформа с индексом {platform_idx} не существует. "
                             f"Доступно платформ: {len(self.platforms)}")
        self.platform = self.platforms[platform_idx]

        # Получаем список устройств на выбранной платформе
        self.devices = self.platform.get_devices()
        if device_idx >= len(self.devices):
            raise IndexError(f"Устройство с индексом {device_idx} не существует на платформе. "
                             f"Доступно устройств: {len(self.devices)}")
        self.device = self.devices[device_idx]

        # Создаём контекст OpenCL для выбранного устройства
        self.ctx = cl.Context([self.device])

        # Создаём командную очередь (с поддержкой профилирования, если нужно)
        self.queue = cl.CommandQueue(self.ctx)

        # Информация о памяти устройства
        self.max_alloc_size = self.device.get_info(cl.device_info.MAX_MEM_ALLOC_SIZE)
        self.global_mem_size = self.device.get_info(cl.device_info.GLOBAL_MEM_SIZE)

        # Кэш для скомпилированных программ (ключ — путь к файлу)
        self.programs: Dict[str, cl.Program] = {}

        print(f"OpenCL инициализирован. Устройство: {self.device.name}")

    def print_platform_info(self):
        """Вывод подробной информации о всех доступных платформах и устройствах."""
        print("\nДоступные платформы и устройства:")
        for i, platform in enumerate(self.platforms):
            print(f"Платформа {i}: {platform.name}")
            print(f"  Вендор: {platform.vendor}")
            print(f"  Версия: {platform.version}")

            devices = platform.get_devices()
            for j, device in enumerate(devices):
                print(f"  Устройство {j}:")
                print(f"    Имя: {device.name}")
                print(f"    Тип: {cl.device_type.to_string(device.type)}")
                print(f"    Макс. размер рабочей группы: {device.max_work_group_size}")
                print(f"    Макс. вычислительные единицы: {device.max_compute_units}")
                print(f"    Глобальная память: {device.global_mem_size / (1024 ** 3):.2f} GB")
                print(f"    Локальная память: {device.local_mem_size / 1024:.2f} KB")
                print(f"    Макс. размер буфера: {device.max_mem_alloc_size / (1024 ** 3):.2f} GB")

    def print_used_device_info(self):
        """Вывод информации об используемом устройстве."""
        print("Используемое устройство:")
        print(f"  Имя: {self.device.name}")
        print(f"  Тип: {cl.device_type.to_string(self.device.type)}")
        print(f"  Макс. размер рабочей группы: {self.device.max_work_group_size}")
        print(f"  Макс. вычислительные единицы: {self.device.max_compute_units}")
        print(f"  Глобальная память: {self.global_mem_size / (1024 ** 3):.2f} GB")
        print(f"  Локальная память: {self.device.local_mem_size / 1024:.2f} KB")
        print(f"  Макс. размер буфера: {self.max_alloc_size / (1024 ** 3):.2f} GB")

        # Дополнительная информация о максимальных размерах рабочего элемента
        try:
            max_work_item_sizes = self.device.get_info(cl.device_info.MAX_WORK_ITEM_SIZES)
            print(f"  Макс. размер рабочего элемента: {max_work_item_sizes}")
        except:
            pass

    def compile_program(self, kernel_path: str) -> cl.Program:
        """
        Компиляция программы OpenCL из файла с исходным кодом.
        Результат кэшируется, повторная компиляция того же файла не производится.

        Параметры
        ----------
        kernel_path : str
            Путь к файлу с кодом ядра (например, './computing_cores/wave2d.cl').

        Возвращает
        ----------
        cl.Program
            Скомпилированная программа OpenCL.

        Исключения
        -----------
        FileNotFoundError
            Если файл не существует.
        RuntimeError
            Если компиляция не удалась.
        """
        if kernel_path in self.programs:
            return self.programs[kernel_path]

        if not os.path.exists(kernel_path):
            raise FileNotFoundError(f"Файл с ядром не найден: {kernel_path}")

        with open(kernel_path, 'r', encoding='utf-8') as f:
            kernel_code = f.read()

        try:
            program = cl.Program(self.ctx, kernel_code).build()
        except Exception as e:
            # Если компиляция не удалась, выводим лог ошибок
            raise RuntimeError(f"Ошибка компиляции OpenCL программы: {e}")

        self.programs[kernel_path] = program
        return program

    def get_kernel(self, program: cl.Program, kernel_name: str) -> cl.Kernel:
        """
        Получение объекта ядра из скомпилированной программы.

        Параметры
        ----------
        program : cl.Program
            Скомпилированная программа OpenCL.
        kernel_name : str
            Имя функции ядра в исходном коде.

        Возвращает
        ----------
        cl.Kernel
            Объект ядра для использования в очереди команд.
        """
        return getattr(program, kernel_name)

    def get_device_memory_info(self) -> Tuple[int, int]:
        """
        Возвращает информацию о памяти устройства.

        Возвращает
        ----------
        tuple (max_alloc_size, global_mem_size)
            Максимальный размер одного выделения и общий объём глобальной памяти в байтах.
        """
        return self.max_alloc_size, self.global_mem_size

    def release_programs(self):
        """Освобождение всех скомпилированных программ (опционально)."""
        self.programs.clear()

    def __del__(self):
        """Деструктор: освобождение ресурсов (очередь и контекст)."""
        if hasattr(self, 'queue') and self.queue:
            self.queue.finish()
        # Явно освобождать контекст не обязательно, но для порядка:
        if hasattr(self, 'ctx') and self.ctx:
            self.ctx = None