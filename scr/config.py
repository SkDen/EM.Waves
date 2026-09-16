import numpy as np

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional, Union, Callable


class BoundaryType(Enum):
    """Типы граничных условий"""
    DIRICHLET = 'dirichlet'
    NEUMANN =   'neumann'
    PERIODIC =  'periodic'
    PML =       'pml'
    WL =        'wl'
    ABSORBING = 'absorbing'


@dataclass
class SimulationConfig:
    """Конфигурация моделирования"""
    # Размеры сетки (внутренние узлы)
    nx: int
    ny: int

    # Шаги по пространству и времени
    dx: float
    dy: float
    dt: float
    
    # Параметры среды
    c: Union[float, np.ndarray]  # скорость волны (скаляр или 2D массив)
    
    # Граничные условия для каждой стороны
    bc: Dict[str, BoundaryType] = None  # ключи: 'left', 'right', 'bottom', 'top'
    
    # Источник (функция f(x,y,t) или массив)
    source: Optional[Union[Callable, np.ndarray]] = None
    use_source: bool = False
    
    # Толщина фиктивного слоя
    d: int = 1

    # Скорость волны в фиктивном заводящем слое
    c_wl: float = 0.5

    # Параметры вывода
    snapshot_interval: int = 100
    output_dir: str = './snapshots'
    
    # Точность
    dtype: type = np.float32
    
    # Проверка устойчивости (CFL)
    def check_cfl(self) -> bool:
        c_max = self.c if np.isscalar(self.c) else np.max(self.c)
        cfl = c_max * self.dt * np.sqrt(1/self.dx**2 + 1/self.dy**2)
        return cfl <= 1.0