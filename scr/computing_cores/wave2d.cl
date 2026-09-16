// Ядра OpenCL для моделирования двумерного волнового уравнения
// Явная конечно-разностная схема с девятиточечным шаблоном
// Уравнение: d^2u/dt^2 = c^2 * (d^2u/dx^2 + d^2u/dy^2) + f

// --------------------------------------------------------------------------
// Основное ядро: вычисление следующего временного слоя
// --------------------------------------------------------------------------
__kernel void update_field(
    __global const float*   u_curr,      // текущий слой (в момент t)
    __global const float*   u_prev,      // предыдущий слой (в момент t-dt)
    __global float*         u_next,      // следующий слой (в момент t+dt) - результат
    __global const float*   c_mass,      // массив скорости среды в узле
    __global const float*   f,           // источник (внешняя сила), того же размера
    float dx,                            // шаг сетки по X (предполагается равным dy)
    float dy,                            // шаг сетки по Y (должен совпадать с dx для 9-точечной схемы)
    float dt,                            // шаг по времени
    float total_time,                    // не используется, оставлен для совместимости
    int nx_full,                         // ширина строки в буферах (с учётом фиктивных границ)
    int ny_full,                         // высота столбца в буферах (с учётом фиктивных границ)
    int d,                               // толщина фиктивного слоя
    int nx,                              // число внутренних узлов по X
    int ny)                              // число внутренних узлов по Y
{
    // Глобальные идентификаторы соответствуют внутренним узлам (0..nx-1, 0..ny-1)
    int i = get_global_id(0);
    int j = get_global_id(1);

    if (i >= nx || j >= ny) return;

    // Линейный индекс центрального узла с учётом фиктивных границ
    int idx = (j + d) * nx_full + (i + d);

    // Загрузка значений в центральном узле и его соседях
    float u_c = u_curr[idx];

    // Ортогональные соседи
    float u_l = u_curr[idx - 1];                // левый
    float u_r = u_curr[idx + 1];                // правый
    float u_d = u_curr[idx - nx_full];          // нижний
    float u_u = u_curr[idx + nx_full];          // верхний

    // Диагональные соседи
    float u_ld = u_curr[idx - nx_full - 1];     // левый-нижний
    float u_rd = u_curr[idx - nx_full + 1];     // правый-нижний
    float u_lu = u_curr[idx + nx_full - 1];     // левый-верхний
    float u_ru = u_curr[idx + nx_full + 1];     // правый-верхний

    // Параметры сетки (предполагаем dx = dy = h)
    float h2 = dx * dx;

    // Стандартная 9-точечная аппроксимация лапласиана (4-й порядок точности)
    // du = ( 1/(6*h2))*[4 * (u_l + u_r + u_d + u_u) + (u_ld + u_rd + u_lu + u_ru) - 20 * u_c ]
    float laplacian = (4.0f * (u_l + u_r + u_d + u_u) +
                              (u_ld + u_rd + u_lu + u_ru) -
                            20.0f * u_c) / (6.0f * h2);

    // Квадрат скорости в данной точке
    float c = c_mass[idx];
    float c2 = c * c;

    // Источник (внешняя сила)
    float source = f[idx];

    if ( (j >= nx_full/2) && (j <= nx_full/2) && (i == ny_full/2)) {
        source = 5.f * sin(2.f/4.f * M_PI * total_time);
    }

    // Временной множитель (dt2)
    float dt2 = dt * dt;

    // Явная схема "скачок" (leapfrog) для волнового уравнения
    u_next[idx] = 2.0f * u_c - u_prev[idx] + c2 * dt2 * laplacian + dt2 * source;
}

// --------------------------------------------------------------------------
// Ядра для граничных условий
// Для каждой стороны своё ядро для обновления фиктивных ячеек.
// Предполагается, что они вызываются с глобальным размером, равным числу узлов вдоль границы.
// Аргумент value интерпретируется следующим образом:
//   если value == 0.0f — устанавливаем фиктивную ячейку в 0 (Dirichlet с нулём);
//   иначе — копируем значение из соседней внутренней ячейки (Neumann).
// --------------------------------------------------------------------------

// Левая граница: i = 0 (фиктивная ячейка с индексом i = -1)
__kernel void apply_bc_left(
    __global float* u,
    int d,
    int nx_full,
    int ny_full,
    float value)
{
    int j = get_global_id(0);  // индекс вдоль границы (0..ny-1)
    // Индекс фиктивной ячейки (i = -1)
    int idx_fict = (j + d) * nx_full + d - 1;
    // Индекс соседней внутренней ячейки (i = 0)
    int idx_inner = idx_fict + 1;

    if (value == 0.0f) {
        u[idx_fict] = 0.0f;
    } else {
        u[idx_fict] = u[idx_inner];
    }
}

// Правая граница: i = nx (фиктивная ячейка с индексом i = nx+1)
__kernel void apply_bc_right(
    __global float* u,
    int d,
    int nx_full,
    int ny_full,
    float value)
{
    int j = get_global_id(0);
    // Индекс фиктивной ячейки (i = nx+1)
    int idx_fict = (j + d) * nx_full + (nx_full - d);  // nx_full, последний индекс
    // Индекс соседней внутренней ячейки (i = nx)
    int idx_inner = idx_fict - 1;

    if (value == 0.0f) {
        u[idx_fict] = 0.0f;
    } else {
        u[idx_fict] = u[idx_inner];
    }
}

// Верхняя граница:
__kernel void apply_bc_bottom(
    __global float* u,
    int d,
    int nx_full,
    int ny_full,
    float value)
{
    int i = get_global_id(0);  // 0..nx-1
    // Индекс фиктивной ячейки
    int idx_fict = (d - 1) * nx_full + i + d;
    // Индекс соседней внутренней ячейки (j = 0)
    int idx_inner = idx_fict + nx_full;

    if (value == 0.0f) {
        u[idx_fict] = 0.0f;
    } else {
        u[idx_fict] = u[idx_inner];
    }
}

// нижняя граница
__kernel void apply_bc_top(
    __global float* u,
    int d,
    int nx_full,
    int ny_full,
    float value)
{
    int i = get_global_id(0);
    // Индекс фиктивной ячейки (j = ny+1)
    int idx_fict = (ny_full - d) * nx_full + (i + d);
    // Индекс соседней внутренней ячейки (j = ny)
    int idx_inner = idx_fict - nx_full;

    if (value == 0.0f) {
        u[idx_fict] = 0.0f;
    } else {
        u[idx_fict] = u[idx_inner];
    }
}

// --------------------------------------------------------------------------
// Поглощяющие граничные условия-ядра
// --------------------------------------------------------------------------
__kernel void apply_absorbing_left(
    __global const float* u_curr,
    __global const float* u_prev,
    __global float*       u_next,
    __global const float* c_mass,
    float dx, float dy, float dt,
    int nx_full, int ny_full, int d,
    int nx, int ny)
{
    int j = get_global_id(0);
    if (j >= ny) return;

    int idx = (j + d) * nx_full + d;   // граничный узел i=0
    float c = c_mass[idx];
    float factor = c * dt / (6.0f * dx);

    float u0 = u_curr[idx];
    float u1 = u_curr[idx + 1];

    // соседи по y на границе
    float u0_up   = u_curr[idx + nx_full];
    float u0_down = u_curr[idx - nx_full];
    float u1_up   = u_curr[idx + nx_full + 1];
    float u1_down = u_curr[idx - nx_full + 1];

    float term_orth = 4.0f * (u1 - u0);
    float term_diag = (u1_up - u0_up) + (u1_down - u0_down);

    u_next[idx] = u0 + factor * (term_orth + term_diag);
}

// Ядро для правой границы (i = nx-1)
__kernel void apply_absorbing_right(
    __global const float* u_curr,
    __global const float* u_prev,
    __global float*       u_next,
    __global const float* c_mass,
    float dx, float dy, float dt,
    int nx_full, int ny_full, int d,
    int nx, int ny)
{
    int j = get_global_id(0);
    if (j >= ny) return;

    int idx = (j + d) * nx_full + (nx + d - 1); // i=nx-1
    float c = c_mass[idx];
    float factor = c * dt / (6.0f * dx);

    float u0 = u_curr[idx];
    float u1 = u_curr[idx - 1];   // сосед слева

    float u0_up   = u_curr[idx + nx_full];
    float u0_down = u_curr[idx - nx_full];
    float u1_up   = u_curr[idx + nx_full - 1];
    float u1_down = u_curr[idx - nx_full - 1];

    float term_orth = 4.0f * (u0 - u1);
    float term_diag = (u0_up - u1_up) + (u0_down - u1_down);

    u_next[idx] = u0 - factor * (term_orth + term_diag);
}

// Ядро для нижней границы (j = 0)
__kernel void apply_absorbing_bottom(
    __global const float* u_curr,
    __global const float* u_prev,
    __global float*       u_next,
    __global const float* c_mass,
    float dx, float dy, float dt,
    int nx_full, int ny_full, int d,
    int nx, int ny)
{
    int i = get_global_id(0);
    if (i >= nx) return;

    int idx = d * nx_full + (i + d);   // j=0
    float c = c_mass[idx];
    float factor = c * dt / (6.0f * dy);

    float u0 = u_curr[idx];
    float u1 = u_curr[idx + nx_full];   // сосед сверху

    // соседи по x на границе
    float u0_right = u_curr[idx + 1];
    float u0_left  = u_curr[idx - 1];
    float u1_right = u_curr[idx + nx_full + 1];
    float u1_left  = u_curr[idx + nx_full - 1];

    float term_orth = 4.0f * (u1 - u0);
    float term_diag = (u1_right - u0_right) + (u1_left - u0_left);

    u_next[idx] = u0 + factor * (term_orth + term_diag);
}

// Ядро для верхней границы (j = ny-1)
__kernel void apply_absorbing_top(
    __global const float* u_curr,
    __global const float* u_prev,
    __global float*       u_next,
    __global const float* c_mass,
    float dx, float dy, float dt,
    int nx_full, int ny_full, int d,
    int nx, int ny)
{
    int i = get_global_id(0);
    if (i >= nx) return;

    int idx = (ny - 1 + d) * nx_full + (i + d); // j=ny-1
    float c = c_mass[idx];
    float factor = c * dt / (6.0f * dy);

    float u0 = u_curr[idx];
    float u1 = u_curr[idx - nx_full];   // сосед снизу

    float u0_right = u_curr[idx + 1];
    float u0_left  = u_curr[idx - 1];
    float u1_right = u_curr[idx - nx_full + 1];
    float u1_left  = u_curr[idx - nx_full - 1];

    float term_orth = 4.0f * (u0 - u1);
    float term_diag = (u0_right - u1_right) + (u0_left - u1_left);

    u_next[idx] = u0 - factor * (term_orth + term_diag);
}

// Ядро для углов
__kernel void apply_absorbing_corners(
    __global const float* u_curr,
    __global const float* u_prev,
    __global float*       u_next,
    __global const float* c_mass,
    float dx, float dy, float dt,
    int nx_full, int ny_full, int d,
    int nx, int ny)
{
    int corner = get_global_id(0);
    if (corner >= 4) return;

    int i, j;
    switch (corner) {
        case 0: i = 0;      j = 0;      break; // левый нижний
        case 1: i = nx-1;   j = 0;      break; // правый нижний
        case 2: i = 0;      j = ny-1;   break; // левый верхний
        case 3: i = nx-1;   j = ny-1;   break; // правый верхний
    }

    int idx = (j + d) * nx_full + (i + d);
    float c = c_mass[idx];
    float u0 = u_curr[idx];
    float factor_x = c * dt / (6.0f * dx);
    float factor_y = c * dt / (6.0f * dy);

    float term_x = 0.0f, term_y = 0.0f;

    // Левая или правая граница с учётом доступных узлов
    if (i == 0) {
        float u1 = u_curr[idx + 1];
        float u0_up = (j < ny-1) ? u_curr[idx + nx_full] : 0.0f;
        float u1_up = (j < ny-1) ? u_curr[idx + nx_full + 1] : 0.0f;
        float u0_down = (j > 0) ? u_curr[idx - nx_full] : 0.0f;
        float u1_down = (j > 0) ? u_curr[idx - nx_full + 1] : 0.0f;
        term_x = 4.0f * (u1 - u0) + (u1_up - u0_up) + (u1_down - u0_down);
    } else if (i == nx-1) {
        float u1 = u_curr[idx - 1];
        float u0_up = (j < ny-1) ? u_curr[idx + nx_full] : 0.0f;
        float u1_up = (j < ny-1) ? u_curr[idx + nx_full - 1] : 0.0f;
        float u0_down = (j > 0) ? u_curr[idx - nx_full] : 0.0f;
        float u1_down = (j > 0) ? u_curr[idx - nx_full - 1] : 0.0f;
        term_x = 4.0f * (u0 - u1) + (u0_up - u1_up) + (u0_down - u1_down);
    }

    // Аналогично для y
    if (j == 0) {
        float u1 = u_curr[idx + nx_full];
        float u0_right = (i < nx-1) ? u_curr[idx + 1] : 0.0f;
        float u1_right = (i < nx-1) ? u_curr[idx + nx_full + 1] : 0.0f;
        float u0_left = (i > 0) ? u_curr[idx - 1] : 0.0f;
        float u1_left = (i > 0) ? u_curr[idx + nx_full - 1] : 0.0f;
        term_y = 4.0f * (u1 - u0) + (u1_right - u0_right) + (u1_left - u0_left);
    } else if (j == ny-1) {
        float u1 = u_curr[idx - nx_full];
        float u0_right = (i < nx-1) ? u_curr[idx + 1] : 0.0f;
        float u1_right = (i < nx-1) ? u_curr[idx - nx_full + 1] : 0.0f;
        float u0_left = (i > 0) ? u_curr[idx - 1] : 0.0f;
        float u1_left = (i > 0) ? u_curr[idx - nx_full - 1] : 0.0f;
        term_y = 4.0f * (u0 - u1) + (u0_right - u1_right) + (u0_left - u1_left);
    }

    u_next[idx] = u0 + 0.5f * (factor_x * term_x + factor_y * term_y);
}

// --------------------------------------------------------------------------
// Заполняем источник
// --------------------------------------------------------------------------

// __kernel void compute_source(
//     __global float* f,
//     float t,
//     float dx, 
//     float dy,
//     int nx_full, 
//     int ny_full, 
//     int d)
// {
//     int i = get_global_id(0); // индекс по x (с учётом фиктивных)
//     int j = get_global_id(1);
//     if (i >= nx_full || j >= ny_full) return;

//     float x = (i - d) * dx;
//     float y = (j - d) * dy;
//     // Пример функции источника (можно параметризовать через define или константы)
//     float r2 = (x - x0)*(x - x0) + (y - y0)*(y - y0);
//     float val = exp(-r2/(2*sigma*sigma)) * sin(2*M_PI*freq*t);
//     f[j * nx_full + i] = val;
// }



// --------------------------------------------------------------------------
// Демпфирующие ядра
// --------------------------------------------------------------------------

__kernel void update_pml_x(
    __global const float* ux_curr,
    __global       float* ux_next,
    __global const float* vx_curr,
    __global       float* vx_next,
    __global const float* c,
    __global const float* sigma_x,
    float dx, float dt,
    int nx_full, int ny_full, int d,
    int nx, int ny)
{
    int i = get_global_id(0);
    int j = get_global_id(1);
    if (i >= nx || j >= ny) return;

    int idx = (j + d) * nx_full + (i + d);
    float s = sigma_x[idx];
    float c2 = c[idx] * c[idx];

    // Вторая производная по x (3-точечная)
    float u_left  = ux_curr[idx - 1];
    float u_right = ux_curr[idx + 1];
    float d2u_dx2 = (u_left - 2.0f * ux_curr[idx] + u_right) / (dx * dx);

    // Обновление vx (полуцелый слой)
    float v_old = vx_curr[idx];
    float v_new = v_old + dt * (c2 * d2u_dx2 - s * v_old);
    vx_next[idx] = v_new;

    // Обновление ux
    ux_next[idx] = ux_curr[idx] + dt * (v_new - s * ux_curr[idx]);
}


__kernel void update_pml_y(
    __global const float* uy_curr,
    __global       float* uy_next,
    __global const float* vy_curr,
    __global       float* vy_next,
    __global const float* c,
    __global const float* sigma_y,
    float dy, float dt,
    int nx_full, int ny_full, int d,
    int nx, int ny)
{
    int i = get_global_id(0);
    int j = get_global_id(1);
    if (i >= nx || j >= ny) return;

    int idx = (j + d) * nx_full + (i + d);
    float s = sigma_y[idx];
    float c2 = c[idx] * c[idx];

    // Вторая производная по y
    float u_down = uy_curr[idx - nx_full];
    float u_up   = uy_curr[idx + nx_full];
    float d2u_dy2 = (u_down - 2.0f * uy_curr[idx] + u_up) / (dy * dy);

    float v_old = vy_curr[idx];
    float v_new = v_old + dt * (c2 * d2u_dy2 - s * v_old);
    vy_next[idx] = v_new;

    uy_next[idx] = uy_curr[idx] + dt * (v_new - s * uy_curr[idx]);
}

__kernel void sum_fields(
    __global const float* ux,
    __global const float* uy,
    __global       float* u,
    int nx_full, int ny_full, int d,
    int nx, int ny)
{
    int i = get_global_id(0);
    int j = get_global_id(1);
    if (i >= nx || j >= ny) return;

    int idx = (j + d) * nx_full + (i + d);
    u[idx] = ux[idx] + uy[idx];
}

// --------------------------------------------------------------------------
// Ядра для периодических граничных условий
// --------------------------------------------------------------------------



// --------------------------------------------------------------------------
// Заводящие ядра
// --------------------------------------------------------------------------



// --------------------------------------------------------------------------
// PML ядра
// --------------------------------------------------------------------------



