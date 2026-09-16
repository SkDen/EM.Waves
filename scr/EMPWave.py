
import numpy as np
import pyopencl as cl
from pyopencl import mem_flags as mf
import os
import time
from typing import Optional, Callable, Union, Dict, Any

from config import SimulationConfig, BoundaryType
from kernel_loader import OpenCLKernelLoader


class Wave2D_GPU:
    """
    Моделирование двумерного волнового уравнения на GPU с использованием
    явной конечно-разностной схемы (девятиточечный шаблон).

    Уравнение: d²u/dt² = c² ∇²u + f(x,y,t)

    Поля u хранятся в трёх временных слоях (предыдущий, текущий, следующий)
    с добавлением фиктивных ячеек для упрощения граничных условий.
    """

    def __init__(self, config: SimulationConfig):
        """
        Инициализация параметров моделирования и создание массивов на хосте.

        Параметры
        ----------
        config : SimulationConfig
            Конфигурация моделирования (размеры сетки, шаги, параметры среды,
            граничные условия, источник и т.д.)
        """
        self.config = config
        self.nx = config.nx         # число внутренних узлов по x
        self.ny = config.ny         # число внутренних узлов по y
        self.dx = config.dx         # шаг по x
        self.dy = config.dy         # шаг по y
        self.dt = config.dt         # шаг по времени
        self.d  = config.d          # толщина фиктивного слоя
        self.c_wl = config.c_wl     # скорость волны в замодещем фиктивном слое
        self.dtype = config.dtype   # тип данных (np.float32 или np.float64)

        # Размеры с учётом фиктивных границ (по d ячеек с каждой стороны)
        self.nx_full = self.nx + 2*self.d
        self.ny_full = self.ny + 2*self.d

        # Параметры среды (скорость волны)
        if np.isscalar(config.c):
            self.c_scalar = float(config.c)
            self.c_array = None            # однородная среда
        else:
            # Неоднородная среда: ожидается массив размера (ny_full, nx_full)
            self.c_array = config.c.astype(self.dtype)
            self.c_scalar = None

        # Источник (может быть функцией f(x,y,t) или массивом)
        self.use_source = config.source is not None
        self.source_func = config.source   # если задана функция
        self.f_host = None                 # массив источника на хосте (будет создан при необходимости)

        # Поля для предвычисленного источника
        self.precomputed_source = None                            # трёхмерный массив (num_steps, ny_full, nx_full)
        self.use_precomputed_source = True  # флаг использования предвычисленных данных

        # Массивы для трёх временных слоёв на хосте (инициализация нулями)
        self.u_prev = np.zeros((self.ny_full, self.nx_full), dtype=self.dtype)
        self.u_curr = np.zeros((self.ny_full, self.nx_full), dtype=self.dtype)
        self.u_next = np.zeros((self.ny_full, self.nx_full), dtype=self.dtype)

        # Граничные условия
        self.bc = config.bc if config.bc else {
            'left':     BoundaryType.DIRICHLET,
            'right':    BoundaryType.DIRICHLET,
            'bottom':   BoundaryType.DIRICHLET,
            'top':      BoundaryType.DIRICHLET
        }

        # OpenCL атрибуты (будут инициализированы позже)
        self.cl_loader: Optional[OpenCLKernelLoader] = None
        self.ctx:       Optional[cl.Context] = None
        self.queue:     Optional[cl.CommandQueue] = None
        self.program:   Optional[cl.Program] = None
        self.kernels:   Dict[str, cl.Kernel] = {}   # словарь скомпилированных ядер
        self.buffers:   Dict[str, cl.Buffer] = {}   # буферы на устройстве

        #.///// Тестовый /////////////////////////////////////////////////
        # Параетры PML граничных слооев
        self.pml_width = 15          # толщина слоя в узлах
        self.pml_power = 2           # степень полинома (2 или 3)
        self.pml_R = 1e-6            # желаемый коэффициент отражения
        #.////////////////////////////////////////////////////////////////

        # Счётчик шагов
        self.step_count = 0

        # время моделирования среды (собственное время моделирования)
        self.total_time = 0

        # Проверка условия устойчивости Куранта
        if not config.check_cfl():
            print("Предупреждение: условие Куранта не выполнено! "
                  "Возможна неустойчивость решения.")

    # ----------------------------------------------------------------------
    # Инициализация OpenCL и управление памятью
    # ----------------------------------------------------------------------

    def setup_opencl(self, 
                     platform_idx: int = 0,
                     device_idx: int = 0,
                     kernel_path: str = 'scr/computing_cores/wave2d.cl'):
        """
        Инициализация OpenCL: выбор устройства, создание контекста и очереди,
        компиляция ядер, создание буферов и копирование начальных данных.

        Параметры
        ----------
        platform_idx : int
            Индекс платформы OpenCL.
        device_idx : int
            Индекс устройства на платформе.
        kernel_path : str
            Путь к файлу с исходным кодом ядер OpenCL.
        """
        # Загрузчик ядер
        self.cl_loader = OpenCLKernelLoader(platform_idx, device_idx)
        self.ctx = self.cl_loader.ctx
        self.queue = self.cl_loader.queue

        # Оценка требуемой памяти и проверка доступности
        required_mem = self._estimate_device_memory()
        if required_mem > self.cl_loader.global_mem_size:
            raise MemoryError(
                f"Требуемая видеопамять {required_mem/(1024**3):.2f} GB превышает доступную "
                f"{self.cl_loader.global_mem_size/(1024**3):.2f} GB.\n"
                "Уменьшите размеры сетки или используйте dtype=np.float32."
            )

        # Компиляция программы
        self.program = self.cl_loader.compile_program(kernel_path)

        # Получение ядер по именам (предполагается, что они определены в wave2d.cl)
        # Основное [рабочее] ядро
        self.kernels['update'] =    self.cl_loader.get_kernel(self.program, 'update_field')
        
        # Ядра для граничных условий Дирихле и Неймана
        self.kernels['bc_left'] =   self.cl_loader.get_kernel(self.program, 'apply_bc_left')
        self.kernels['bc_right'] =  self.cl_loader.get_kernel(self.program, 'apply_bc_right')
        self.kernels['bc_bottom'] = self.cl_loader.get_kernel(self.program, 'apply_bc_bottom')
        self.kernels['bc_top'] =    self.cl_loader.get_kernel(self.program, 'apply_bc_top')
        
        # Поглощяющие ядра
        self.kernels['absorbing_left'] =    self.cl_loader.get_kernel(self.program, 'apply_absorbing_left')
        self.kernels['absorbing_right'] =   self.cl_loader.get_kernel(self.program, 'apply_absorbing_right')
        self.kernels['absorbing_bottom'] =  self.cl_loader.get_kernel(self.program, 'apply_absorbing_bottom')
        self.kernels['absorbing_top'] =     self.cl_loader.get_kernel(self.program, 'apply_absorbing_top')
        self.kernels['absorbing_corners'] = self.cl_loader.get_kernel(self.program, 'apply_absorbing_corners')

        #.///// Тестовый ////////////////////////////////////////////////////////

        self.kernels['update_pml_x'] = self.cl_loader.get_kernel(self.program, 'update_pml_x')
        self.kernels['update_pml_y'] = self.cl_loader.get_kernel(self.program, 'update_pml_y')
        self.kernels['sum_fields']   = self.cl_loader.get_kernel(self.program, 'sum_fields')
        self.kernels['add_source']   = self.cl_loader.get_kernel(self.program, 'add_source')
        self.kernels['redistribute'] = self.cl_loader.get_kernel(self.program, 'redistribute')

        #.///////////////////////////////////////////////////////////////////////

        # добавить ядра для периодических границ и PML

        # Создание буферов на устройстве
        mf = cl.mem_flags
        self.buffers['u_prev'] = cl.Buffer(self.ctx, mf.READ_WRITE, self.u_prev.nbytes)
        self.buffers['u_curr'] = cl.Buffer(self.ctx, mf.READ_WRITE, self.u_curr.nbytes)
        self.buffers['u_next'] = cl.Buffer(self.ctx, mf.WRITE_ONLY, self.u_next.nbytes)


        #.///// Тестовый ////////////////////////////////////////////////////////

        # Буферы для PML
        size = self.nx_full * self.ny_full * self.dtype.itemsize
        self.buffers['ux_prev'] = cl.Buffer(self.ctx, mf.READ_WRITE, size)
        self.buffers['ux_curr'] = cl.Buffer(self.ctx, mf.READ_WRITE, size)
        self.buffers['ux_next'] = cl.Buffer(self.ctx, mf.WRITE_ONLY, size)

        self.buffers['uy_prev'] = cl.Buffer(self.ctx, mf.READ_WRITE, size)
        self.buffers['uy_curr'] = cl.Buffer(self.ctx, mf.READ_WRITE, size)
        self.buffers['uy_next'] = cl.Buffer(self.ctx, mf.WRITE_ONLY, size)

        self.buffers['vx_curr'] = cl.Buffer(self.ctx, mf.READ_WRITE, size)
        self.buffers['vx_next'] = cl.Buffer(self.ctx, mf.WRITE_ONLY, size)
        self.buffers['vy_curr'] = cl.Buffer(self.ctx, mf.READ_WRITE, size)
        self.buffers['vy_next'] = cl.Buffer(self.ctx, mf.WRITE_ONLY, size)

        # Профили затухания
        sigma_x, sigma_y = self._create_pml_profiles()
        self.buffers['sigma_x'] = cl.Buffer(self.ctx, mf.READ_ONLY, sigma_x.nbytes)
        self.buffers['sigma_y'] = cl.Buffer(self.ctx, mf.READ_ONLY, sigma_y.nbytes)
        cl.enqueue_copy(self.queue, self.buffers['sigma_x'], sigma_x)
        cl.enqueue_copy(self.queue, self.buffers['sigma_y'], sigma_y)

        # Инициализация расщеплённых полей нулями (позже будут перезаписаны начальными условиями)
        zero_host = np.zeros((self.ny_full, self.nx_full), dtype=self.dtype)
        for buf in ['ux_prev', 'ux_curr', 'ux_next', 'uy_prev', 'uy_curr', 'uy_next',
                    'vx_curr', 'vx_next', 'vy_curr', 'vy_next']:
            cl.enqueue_copy(self.queue, self.buffers[buf], zero_host)

        #.///////////////////////////////////////////////////////////////////////

        # Параметры среды
        if self.c_array is not None:
            c_host = self.c_array
        else:
            # Создаём массив, заполненный константой
            c_host = np.full((self.ny_full, self.nx_full), self.c_scalar, dtype=self.dtype)

        self.buffers['c'] = cl.Buffer(self.ctx, mf.READ_ONLY, c_host.nbytes)
        cl.enqueue_copy(self.queue, self.buffers['c'], c_host)

        # Буфер для источника (если он задан)
        self.f_host = np.zeros((self.ny_full, self.nx_full), dtype=self.dtype)
        self.buffers['f'] = cl.Buffer(self.ctx, mf.READ_ONLY, self.f_host.nbytes)
        cl.enqueue_copy(self.queue, self.buffers['f'], self.f_host)


        # Копирование начальных данных полей на устройство
        self._copy_host_to_device()
        print("OpenCL буферы созданы и начальные данные скопированы на устройство.")

    def _estimate_device_memory(self) -> int:
        """
        Оценка необходимого объёма видеопамяти для всех буферов.

        Возвращает
        ----------
        int
            Требуемый объём в байтах.
        """
        item_size = np.dtype(self.dtype).itemsize
        size_per_array = self.nx_full * self.ny_full * item_size
        total = 3 * size_per_array                     # три слоя u
        if self.c_array is not None:
            total += size_per_array                     # массив скорости
        if self.source_func is not None:
            total += size_per_array                     # массив источника
        return total

    def _copy_host_to_device(self):
        """Копирование массивов с хоста в буферы устройства."""
        cl.enqueue_copy(self.queue, self.buffers['u_prev'], self.u_prev)
        cl.enqueue_copy(self.queue, self.buffers['u_curr'], self.u_curr)
        # u_next не копируется, так как будет перезаписан на первом шаге


        if 'f' in self.buffers and self.f_host is not None:
            cl.enqueue_copy(self.queue, self.buffers['f'], self.f_host)
        self.queue.finish()

    # ----------------------------------------------------------------------
    # Предычисление массивов временного среза источника (если позволяет ОЗУ)
    # ----------------------------------------------------------------------

    def pre_calc_source(self, num_steps: int, max_memory_mb: float = None):
        """
        Предварительное вычисление всех временных срезов источника.

        Параметры
        ----------
        num_steps : int
            Количество временных шагов, для которых необходимо вычислить источник.
        max_memory_mb : float, optional
            Максимальный допустимый объём оперативной памяти (в МБ) для хранения
            предвычисленных данных. Если не указан, используется значение из конфига
            или 80% от доступной памяти (оценка).

        Возвращает
        ----------
        bool
            True, если предвычисление выполнено, иначе False.
        """
        # 1. Оценка необходимой памяти
        itemsize = np.dtype(self.dtype).itemsize
        required_bytes = num_steps * self.nx_full * self.ny_full * itemsize
        required_mb = required_bytes / (1024 ** 2)

        # 2. Определение допустимого лимита памяти
        if max_memory_mb is None:
            # Если не задано, используем 80% от доступной физической памяти
            import psutil
            available_mb = psutil.virtual_memory().available / (1024 ** 2)
            max_memory_mb = available_mb * 0.8
        else:
            max_memory_mb = float(max_memory_mb)

        # 3. Проверка возможности предвычисления
        if required_mb > max_memory_mb:
            print(f"Предвычисление источника невозможно: требуется {required_mb:.1f} МБ, "
                f"доступно {max_memory_mb:.1f} МБ. Будет использоваться вычисление на лету.")
            self.use_precomputed_source = False
            return False

        # 4. Создание трёхмерного массива и заполнение
        print(f"Предвычисление {num_steps} слоёв источника...")
        self.precomputed_source = np.zeros((num_steps, self.ny_full, self.nx_full),
                                            dtype=self.dtype)

        # Сетка координат для внутренних узлов (без фиктивных границ)
        x = np.linspace(0, (self.nx - 1) * self.dx, self.nx)
        y = np.linspace(0, (self.ny - 1) * self.dy, self.ny)
        X, Y = np.meshgrid(x, y, indexing='ij')
        inner_slice_y = slice(self.d, self.d + self.ny)
        inner_slice_x = slice(self.d, self.d + self.nx)

        # Цикл по временным шагам
        for step in range(num_steps):
            t = step * self.dt
            # Вычисляем значения во внутренних узлах
            inner = self.source_func(X, Y, t).T  # форма (ny, nx)
            # Записываем в предвычисленный массив
            self.precomputed_source[step, inner_slice_y, inner_slice_x] = inner

            # (Опционально) прогресс-бар
            if (step + 1) % max(1, num_steps // 10) == 0:
                print(f"  Прогресс: {step + 1}/{num_steps}")

        self.use_precomputed_source = True
        print("Предвычисление источника завершено.")
        return True

    # ----------------------------------------------------------------------
    # Установка начальных условий
    # ----------------------------------------------------------------------

    def set_initial_condition(self, func: Callable[[float, float], float]):
        """
        Установка начального поля u(x, y, 0) = func(x, y).

        Параметры
        ----------
        func : callable
            Функция, принимающая координаты x, y и возвращающая значение поля.
        """
        # Координаты внутренних узлов (без фиктивных границ)
        x = np.linspace(0, (self.nx - 1) * self.dx, self.nx)
        y = np.linspace(0, (self.ny - 1) * self.dy, self.ny)
        X, Y = np.meshgrid(x, y, indexing='ij')  # X.shape = (nx, ny)

        # Значения во внутренних узлах
        inner = func(X, Y).T  # теперь форма (ny, nx)

        # Срезы для внутренней области с учётом толщины фиктивного слоя
        inner_slice_y = slice(self.d, self.d + self.ny)
        inner_slice_x = slice(self.d, self.d + self.nx)

        # Заполняем оба временных слоя
        self.u_prev[inner_slice_y, inner_slice_x] = inner
        self.u_curr[inner_slice_y, inner_slice_x] = inner

        # Применяем граничные условия к фиктивным ячейкам
        self._apply_boundary_conditions_host(self.u_prev)
        self._apply_boundary_conditions_host(self.u_curr)

        # Если OpenCL уже инициализирован, копируем данные на устройство
        if self.buffers:
            self._copy_host_to_device()

    def set_initial_condition_array(self, arr: np.ndarray):
        """
        Установка начального поля из двумерного массива (без фиктивных границ).

        Параметры
        ----------
        arr : np.ndarray
            Массив размера (ny, nx) с начальными значениями во внутренних узлах.
        """
        if arr.shape != (self.ny, self.nx):
            raise ValueError(f"Ожидаемый размер массива ({self.ny}, {self.nx}), получен {arr.shape}")

        inner_slice_y = slice(self.d, self.d + self.ny)
        inner_slice_x = slice(self.d, self.d + self.nx)

        self.u_prev[inner_slice_y, inner_slice_x] = arr
        self.u_curr[inner_slice_y, inner_slice_x] = arr

        self._apply_boundary_conditions_host(self.u_prev)
        self._apply_boundary_conditions_host(self.u_curr)

        if self.buffers:
            self._copy_host_to_device()

    # ----------------------------------------------------------------------
    # Граничные условия (на хосте и устройстве)
    # расчет параметров граничных условий
    # ----------------------------------------------------------------------
    

    #.////// Тестовый //////////////////////////////////////////////////////////////////////
    def _create_pml_profiles(self):
        nx = self.nx
        ny = self.ny
        nx_full = self.nx_full
        ny_full = self.ny_full
        dx = self.dx
        dy = self.dy
        c_max = np.max(self.c_array) if self.c_array is not None else self.c_scalar

        sigma_x = np.zeros((ny_full, nx_full), dtype=self.dtype)
        sigma_y = np.zeros((ny_full, nx_full), dtype=self.dtype)

        # Расчёт sigma_max
        sigma_max = - (self.pml_power + 1) * c_max * np.log(self.pml_R) / (2 * self.pml_width * dx)

        # Левая и правая границы
        for i in range(self.pml_width):
            val = sigma_max * ((self.pml_width - i) / self.pml_width) ** self.pml_power
            sigma_x[:, i] = val
            sigma_x[:, nx_full - 1 - i] = val

        # Нижняя и верхняя границы
        for j in range(self.pml_width):
            val = sigma_max * ((self.pml_width - j) / self.pml_width) ** self.pml_power
            sigma_y[j, :] = val
            sigma_y[ny_full - 1 - j, :] = val

        return sigma_x, sigma_y
    #.///////////////////////////////////////////////////////////////////////////////////////


    # На данный момент требует даработки переработки или удаления
    def _apply_boundary_conditions_host(self, arr: np.ndarray):
        """
        Применение граничных условий к массиву на хосте (заполнение фиктивных ячеек).

        Параметры
        ----------
        arr : np.ndarray
            Массив размера (ny_full, nx_full), для которого заполняются границы.
        """
        # Левые фиктивные столбцы (x = 0 .. d-1)
        bc_left = self.bc['left']
        if bc_left == BoundaryType.DIRICHLET:
            arr[:, :self.d] = 0.0
        elif bc_left == BoundaryType.NEUMANN:
            # Копируем из ближайшего внутреннего столбца (индекс d)
            arr[:, :self.d] = arr[:, self.d:self.d+1]
        elif bc_left == BoundaryType.PERIODIC:
            # Копируем из правой внутренней области (последние d столбцов внутренней области)
            # Внутренняя область: индексы от d до d+nx-1
            arr[:, :self.d] = arr[:, self.d+self.nx - self.d : self.d+self.nx]

        # Правые фиктивные столбцы (x = nx_full-d .. nx_full-1)
        bc_right = self.bc['right']
        if bc_right == BoundaryType.DIRICHLET:
            arr[:, -self.d:] = 0.0
        elif bc_right == BoundaryType.NEUMANN:
            arr[:, -self.d:] = arr[:, self.d+self.nx - 1 : self.d+self.nx]
        elif bc_right == BoundaryType.PERIODIC:
            arr[:, -self.d:] = arr[:, self.d : self.d+self.d]

        # Нижние фиктивные строки (y = 0 .. d-1)
        bc_bottom = self.bc['bottom']
        if bc_bottom == BoundaryType.DIRICHLET:
            arr[:self.d, :] = 0.0
        elif bc_bottom == BoundaryType.NEUMANN:
            arr[:self.d, :] = arr[self.d:self.d+1, :]
        elif bc_bottom == BoundaryType.PERIODIC:
            arr[:self.d, :] = arr[self.d+self.ny - self.d : self.d+self.ny, :]

        # Верхние фиктивные строки (y = ny_full-d .. ny_full-1)
        bc_top = self.bc['top']
        if bc_top == BoundaryType.DIRICHLET:
            arr[-self.d:, :] = 0.0
        elif bc_top == BoundaryType.NEUMANN:
            arr[-self.d:, :] = arr[self.d+self.ny - 1 : self.d+self.ny, :]
        elif bc_top == BoundaryType.PERIODIC:
            arr[-self.d:, :] = arr[self.d : self.d+self.d, :]

    # ----------------------------------------------------------------------
    # Граничные условия на устройстве
    # ----------------------------------------------------------------------

    def _apply_boundary_to_next(self):
        """
        Применяет граничные условия к u_next (новому слою) после его вычисления.
        Для каждой стороны в зависимости от типа условия:
        - DIRICHLET: обнуление границы и фиктивных узлов.
        - NEUMANN: копирование граничных значений в фиктивные узлы.
        - ABSORBING: сначала Mur на границу, затем копирование в фиктивные узлы.
        Углы обрабатываются отдельно, если хотя бы одна сторона ABSORBING.
        """

        # Аргументы для поглощающих ядер (используем u_curr, u_prev, u_next)
        args_absorb = [
            self.buffers['u_curr'],
            self.buffers['u_prev'],
            self.buffers['u_next'],
            self.buffers['c'],
            np.float32(self.dx),
            np.float32(self.dy),
            np.float32(self.dt),
            np.int32(self.nx_full),
            np.int32(self.ny_full),
            np.int32(self.d),
            np.int32(self.nx),
            np.int32(self.ny)
        ]

        # ---- Левая граница ----
        bc_left = self.bc['left']
        if bc_left == BoundaryType.DIRICHLET:
            # Обнулить фиктивные узлы (i=-1)
            self.kernels['bc_left'](self.queue, (self.ny,), None,
                                    self.buffers['u_next'],
                                    np.int32(self.d),
                                    np.int32(self.nx_full),
                                    np.int32(self.ny_full),
                                    np.float32(0.0))
        elif bc_left == BoundaryType.NEUMANN:
            # Копировать граничные значения в фиктивные узлы
            self.kernels['bc_left'](self.queue, (self.ny,), None,
                                    self.buffers['u_next'],
                                    np.int32(self.d),
                                    np.int32(self.nx_full),
                                    np.int32(self.ny_full),
                                    np.float32(1.0))
        elif bc_left == BoundaryType.ABSORBING:
            # 1) Применить Mur к граничным узлам (i=0)
            self.kernels['absorbing_left'](self.queue, (self.ny,), None, *args_absorb)
            # 2) Копировать новые граничные значения в фиктивные узлы
            self.kernels['bc_left'](self.queue, (self.ny,), None,
                                    self.buffers['u_next'],
                                    np.int32(self.d),
                                    np.int32(self.nx_full),
                                    np.int32(self.ny_full),
                                    np.float32(0.0))

        # ---- Правая граница ----
        bc_right = self.bc['right']
        if bc_right == BoundaryType.DIRICHLET:
            self.kernels['bc_right'](self.queue, (self.ny,), None,
                                    self.buffers['u_next'],
                                    np.int32(self.d),
                                    np.int32(self.nx_full),
                                    np.int32(self.ny_full),
                                    np.float32(0.0))
        elif bc_right == BoundaryType.NEUMANN:
            self.kernels['bc_right'](self.queue, (self.ny,), None,
                                    self.buffers['u_next'],
                                    np.int32(self.d),
                                    np.int32(self.nx_full),
                                    np.int32(self.ny_full),
                                    np.float32(1.0))
        elif bc_right == BoundaryType.ABSORBING:
            self.kernels['absorbing_right'](self.queue, (self.ny,), None, *args_absorb)
            self.kernels['bc_right'](self.queue, (self.ny,), None,
                                    self.buffers['u_next'],
                                    np.int32(self.d),
                                    np.int32(self.nx_full),
                                    np.int32(self.ny_full),
                                    np.float32(0.0))

        # ---- Нижняя граница (j=0) ----
        bc_bottom = self.bc['bottom']
        if bc_bottom == BoundaryType.DIRICHLET:
            self.kernels['bc_bottom'](self.queue, (self.nx,), None,
                                    self.buffers['u_next'],
                                    np.int32(self.d),
                                    np.int32(self.nx_full),
                                    np.int32(self.ny_full),
                                    np.float32(0.0))
        elif bc_bottom == BoundaryType.NEUMANN:
            self.kernels['bc_bottom'](self.queue, (self.nx,), None,
                                    self.buffers['u_next'],
                                    np.int32(self.d),
                                    np.int32(self.nx_full),
                                    np.int32(self.ny_full),
                                    np.float32(1.0))
        elif bc_bottom == BoundaryType.ABSORBING:
            self.kernels['absorbing_bottom'](self.queue, (self.nx,), None, *args_absorb)
            self.kernels['bc_bottom'](self.queue, (self.nx,), None,
                                    self.buffers['u_next'],
                                    np.int32(self.d),
                                    np.int32(self.nx_full),
                                    np.int32(self.ny_full),
                                    np.float32(0.0))

        # ---- Верхняя граница (j=ny-1) ----
        bc_top = self.bc['top']
        if bc_top == BoundaryType.DIRICHLET:
            self.kernels['bc_top'](self.queue, (self.nx,), None,
                                self.buffers['u_next'],
                                np.int32(self.d),
                                np.int32(self.nx_full),
                                np.int32(self.ny_full),
                                np.float32(0.0))
        elif bc_top == BoundaryType.NEUMANN:
            self.kernels['bc_top'](self.queue, (self.nx,), None,
                                self.buffers['u_next'],
                                np.int32(self.d),
                                np.int32(self.nx_full),
                                np.int32(self.ny_full),
                                np.float32(1.0))
        elif bc_top == BoundaryType.ABSORBING:
            self.kernels['absorbing_top'](self.queue, (self.nx,), None, *args_absorb)
            self.kernels['bc_top'](self.queue, (self.nx,), None,
                                self.buffers['u_next'],
                                np.int32(self.d),
                                np.int32(self.nx_full),
                                np.int32(self.ny_full),
                                np.float32(0.0))

        # ---- Углы (только если есть поглощающие границы) ----
        if any(bc == BoundaryType.ABSORBING for bc in self.bc.values()):
            self.kernels['absorbing_corners'](self.queue, (4,), None, *args_absorb)

        self.queue.finish()

    # ----------------------------------------------------------------------
    # Установка источника на хосте
    # ----------------------------------------------------------------------

    def _update_source(self, step_idx: int):
        """
        Обновление буфера источника на устройстве.

        Параметры
        ----------
        step_idx : int
            Номер текущего шага (индекс в предвычисленном массиве, если используется).
        """
        if not self.use_source:
            return

        if self.use_precomputed_source:
            # Берём готовый срез из предвычисленного массива
            if step_idx < len(self.precomputed_source):
                # Копируем весь срез (включая фиктивные ячейки) в f_host
                self.f_host[:] = self.precomputed_source[step_idx]
            else:
                # В случае если массивов состояния источника больше нет
                # просто приводим пустой слой источника
                self.f_host[:] = 0
        else:
            # Старый способ: вычисляем функцию на лету
            x = np.linspace(0, (self.nx - 1) * self.dx, self.nx)
            y = np.linspace(0, (self.ny - 1) * self.dy, self.ny)
            X, Y = np.meshgrid(x, y, indexing='ij')
            inner = self.source_func(X, Y, self.total_time).T
            inner_slice_y = slice(self.d, self.d + self.ny)
            inner_slice_x = slice(self.d, self.d + self.nx)
            self.f_host[inner_slice_y, inner_slice_x] = inner

        # Копирование на устройство (всегда)
        cl.enqueue_copy(self.queue, self.buffers['f'], self.f_host)

    # ----------------------------------------------------------------------
    # Основные методы вычислений
    # ----------------------------------------------------------------------

    def step(self):
        """
        Выполнить один временной шаг.
        """
        # 1. Вычисление u_next для всех внутренних узлов
        args = [
            self.buffers['u_curr'],
            self.buffers['u_prev'],
            self.buffers['u_next'],
            self.buffers['c'],
            self.buffers['f'],
            np.float32(self.dx),
            np.float32(self.dy),
            np.float32(self.dt),
            np.float32(self.total_time),
            np.int32(self.nx_full),
            np.int32(self.ny_full),
            np.int32(self.d),
            np.int32(self.nx),
            np.int32(self.ny)
        ]
        self.kernels['update'](self.queue, (self.nx, self.ny), None, *args)

        # 2. Применение граничных условий к u_next
        self._apply_boundary_to_next()

        # 3. Циклическая перестановка буферов
        self.buffers['u_prev'], self.buffers['u_curr'], self.buffers['u_next'] = \
            self.buffers['u_curr'], self.buffers['u_next'], self.buffers['u_prev']

        self.step_count += 1
        self.total_time += self.dt


    def run(self, num_steps: int, snapshot_interval: Optional[int] = None,
            output_dir: Optional[str] = None):
        """
        Запуск моделирования на заданное число шагов с периодическим сохранением.

        Параметры
        ----------
        num_steps : int
            Количество временных шагов.
        snapshot_interval : int, optional
            Интервал сохранения снимков. Если не указан, используется значение из конфига.
        output_dir : str, optional
            Директория для сохранения снимков. Если не указана, используется из конфига.
        """
        if snapshot_interval is None:
            snapshot_interval = self.config.snapshot_interval
        if output_dir is None:
            output_dir = self.config.output_dir

        os.makedirs(output_dir, exist_ok=True)

        start_time = time.time()
        for step in range(1, num_steps + 1):
            self.step()

            if step % snapshot_interval == 0:
                self.save_snapshot(step, output_dir)
                elapsed = time.time() - start_time
                print(f"Шаг {step}/{num_steps} выполнен, время: {elapsed:.2f} с")

        total_time = time.time() - start_time
        print(f"Моделирование завершено за {total_time:.2f} с")

    # ----------------------------------------------------------------------
    # Сохранение результатов
    # ----------------------------------------------------------------------

    def save_snapshot(self, step: int, output_dir: str):
        """
        Сохранить текущее поле (слой u_curr) в файл формата .npy.
        Извлекается только внутренняя область (без фиктивных границ).

        Параметры
        ----------
        step : int
            Номер шага (используется в имени файла).
        output_dir : str
            Директория для сохранения.
        """
        # Копируем текущий слой с устройства на хост
        cl.enqueue_copy(self.queue, self.u_curr, self.buffers['u_curr'])
        self.queue.finish()

        # Извлекаем внутреннюю область
        snapshot = self.u_curr[1:self.ny+1, 1:self.nx+1].copy()

        # Сохраняем в файл
        filename = os.path.join(output_dir, f"snapshot_{step:06d}.npy")
        np.save(filename, snapshot)

    # ----------------------------------------------------------------------
    # Освобождение ресурсов
    # ----------------------------------------------------------------------

    def free_resources(self):
        """Освобождение всех буферов OpenCL и контекста."""
        for buf in self.buffers.values():
            if isinstance(buf, cl.Buffer):
                buf.release()
        self.buffers.clear()
        if self.ctx:
            self.ctx = None
        print("Ресурсы OpenCL освобождены.")

    def __del__(self):
        """Деструктор для автоматического освобождения ресурсов."""
        self.free_resources()