import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# 1. ГЕОМЕТРИЯ (11x4 сетка, 110 мм ширина, 16 мм высота)

W = 110.0
H = 16.0
n_x = 11
n_y = 4

# Генерируем узлы
nodes = []
for j in range(n_y + 1):
    for i in range(n_x + 1):
        nodes.append([i * W / n_x, j * H / n_y])
nodes = np.array(nodes)

# Удаляем узлы только на верхнем ряду (Y=16), чтобы вырезать коррозию
remove_mask = np.zeros(len(nodes), dtype=bool)
for idx, (x, y) in enumerate(nodes):
    if y == 16:
        # X = 50 / X = 60 (центр сверху) - вырезаем коррозию
        if x == 50 or x == 60:
            remove_mask[idx] = True

nodes = nodes[~remove_mask]

# 2. АВТОМАТИЧЕСКАЯ СОЕДИНЕНИЕ (Триангуляция)

elements = []
n_nodes = len(nodes)

dx = W / n_x + 0.1 #максимальные расстояния между узлами, которые ещё считаются соседями
dy = H / n_y + 0.1

for i in range(n_nodes):
    for j in range(i + 1, n_nodes):
        x1, y1 = nodes[i]
        x2, y2 = nodes[j]
        if abs(x1 - x2) <= dx and abs(y1 - y2) <= dy:
            for k in range(j + 1, n_nodes):
                x3, y3 = nodes[k]
                if abs(x1 - x3) <= dx and abs(y1 - y3) <= dy and abs(x2 - x3) <= dx and abs(y2 - y3) <= dy:
                    area = 0.5 * abs((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1))
                    if area > 0.01:
                        # Не добавляем треугольники, содержащие запрещенные связи
                        # Трещина слева: 50-51, 38-39, 26-39, 27-38, 38-51, 39-50
                        if not (50 in [i, j, k] and 51 in [i, j, k]):
                            if not (38 in [i, j, k] and 39 in [i, j, k]):
                                if not (26 in [i, j, k] and 39 in [i, j, k]):
                                    if not (27 in [i, j, k] and 38 in [i, j, k]):
                                        if not (38 in [i, j, k] and 51 in [i, j, k]):
                                            if not (39 in [i, j, k] and 50 in [i, j, k]):
                                                # Трещина справа: 55-56, 45-46, 33-36, 34-45, 45-56, 46-55
                                                if not (55 in [i, j, k] and 56 in [i, j, k]):
                                                    if not (45 in [i, j, k] and 46 in [i, j, k]):
                                                        if not (33 in [i, j, k] and 46 in [i, j, k]):
                                                            if not (34 in [i, j, k] and 45 in [i, j, k]):
                                                                if not (45 in [i, j, k] and 56 in [i, j, k]):
                                                                    if not (46 in [i, j, k] and 55 in [i, j, k]):
                                                                        elements.append([i, j, k])

# Убираем дубликаты
elements = list(set(tuple(sorted(e)) for e in elements))
elements = [list(e) for e in elements]

print(f"Узлов: {len(nodes)}")
print(f"Элементов: {len(elements)}")

# 3. МАТЕРИАЛ И ПАРАМЕТРЫ

E = 200e9       #модуль Юнга для стали
nu = 0.3        #коэффициент Пуассона
t = 1.0         #толщина
force = 100000  #растягивающая сила

# 4. ФУНКЦИИ РАСЧЕТА (МКЭ)

# Локальная матрица
def compute_local_stiffness_triangle(node1, node2, node3, E, nu, t):
    x1, y1 = node1
    x2, y2 = node2
    x3, y3 = node3
    area = 0.5 * abs((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)) #площадь треугольника

# Коэффициенты формы
    b1 = y2 - y3
    b2 = y3 - y1
    b3 = y1 - y2
    c1 = x3 - x2
    c2 = x1 - x3
    c3 = x2 - x1

# Матрица упругости
    D = (E / (1 - nu ** 2)) * np.array([[1, nu, 0], [nu, 1, 0], [0, 0, (1 - nu) / 2]])

# Матрица градиентов
    B = np.zeros((3, 6))
    B[0, 0] = b1
    B[0, 2] = b2
    B[0, 4] = b3
    B[1, 1] = c1
    B[1, 3] = c2
    B[1, 5] = c3
    B[2, 0] = c1
    B[2, 1] = b1
    B[2, 2] = c2
    B[2, 3] = b2
    B[2, 4] = c3
    B[2, 5] = b3

    B = B / (2 * area)

    # Итоговая локальная матрица
    Ke = t * area * (B.T @ D @ B)
    return Ke, area, B, D

# Глобальная матрица
def assemble_global_stiffness(nodes, elements, E, nu, t):
    n_nodes = len(nodes)
    n_dof = 2 * n_nodes
    K_global = np.zeros((n_dof, n_dof))
    dof_map = np.zeros((n_nodes, 2), dtype=int)
    for i in range(n_nodes):
        dof_map[i, 0] = 2 * i
        dof_map[i, 1] = 2 * i + 1

    local_matrices = []
    for elem in elements:
        coords = [nodes[node] for node in elem]
        Ke, area, B, D = compute_local_stiffness_triangle(coords[0], coords[1], coords[2], E, nu, t)
        local_matrices.append({'nodes': elem, 'area': area, 'matrix': Ke, 'B': B, 'D': D})

        for i, node_i in enumerate(elem):
            for j, node_j in enumerate(elem):
                for dof_i in range(2):
                    for dof_j in range(2):
                        global_i = dof_map[node_i, dof_i]
                        global_j = dof_map[node_j, dof_j]
                        K_global[global_i, global_j] += Ke[2 * i + dof_i, 2 * j + dof_j]
    return K_global, dof_map, local_matrices

# Граничные условия
def apply_boundary_conditions(K, F, fixed_dofs):
    K_mod = K.copy();
    F_mod = F.copy()
    for idx in fixed_dofs:
        K_mod[idx, :] = 0;
        K_mod[:, idx] = 0;
        K_mod[idx, idx] = 1;
        F_mod[idx] = 0
    return K_mod, F_mod

# Расчет напряжений
def compute_stresses(local_matrices, U, dof_map):
    stresses = []
    for lm in local_matrices:
        elem = lm['nodes'];
        B = lm['B'];
        D = lm['D']
        u_elem = np.array([U[dof_map[elem[0], 0]], U[dof_map[elem[0], 1]],
                           U[dof_map[elem[1], 0]], U[dof_map[elem[1], 1]],
                           U[dof_map[elem[2], 0]], U[dof_map[elem[2], 1]]])
        epsilon = B @ u_elem;
        sigma = D @ epsilon
        stresses.append({'index': elem, 'sigma_x': sigma[0], 'sigma_y': sigma[1], 'tau_xy': sigma[2]})
    return stresses

# 5. ВИЗУАЛИЗАЦИЯ

def plot_mesh(nodes, elements):
    fig, ax = plt.subplots(figsize=(14, 6))
    for elem in elements:
        tri = patches.Polygon(nodes[elem], closed=True, fill=False, edgecolor='blue', linewidth=1.5)
        ax.add_patch(tri)

    for i, (x, y) in enumerate(nodes):
        ax.plot(x, y, 'ko', markersize=5)
        ax.text(x + 0.5, y + 0.5, str(i), fontsize=8, fontweight='bold')

    ax.set_xlim(-15, 125)
    ax.set_ylim(-5, 25)
    ax.set_aspect('equal')
    ax.set_xlabel("X, мм")
    ax.set_ylabel("Y, мм")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_deformation(nodes, elements, U, stresses, force_nodes, force_value):
    fig, ax = plt.subplots(figsize=(14, 7))
    real_displacements = U.reshape(-1, 2)
    max_disp = np.max(np.abs(real_displacements))
    display_scale = 3 / max_disp if max_disp > 1e-10 else 1
    deformed_nodes = nodes + real_displacements * display_scale

    # Исходная сетка
    for elem in elements:
        ax.add_patch(
            patches.Polygon(nodes[elem], closed=True, fill=False, edgecolor='blue', linewidth=1, linestyle='--',
                            alpha=0.7))
        ax.add_patch(patches.Polygon(deformed_nodes[elem], closed=True, fill=False, edgecolor='red', linewidth=2))

    # Узлы
    ax.plot(nodes[:, 0], nodes[:, 1], 'bo', markersize=6, label='Исходный', alpha=0.7)
    ax.plot(deformed_nodes[:, 0], deformed_nodes[:, 1], 'rs', markersize=6, label='Деформированный')

    # Векторы деформации
    for orig, defo in zip(nodes, deformed_nodes):
        ax.arrow(orig[0], orig[1], (defo[0] - orig[0]), (defo[1] - orig[1]), head_width=0.3, head_length=0.5,
                 fc='green', ec='green', alpha=0.7)

    # Отображение сил
    for node_idx in force_nodes:
        node_pos = nodes[node_idx]
        if node_pos[0] == 0:
            ax.arrow(node_pos[0], node_pos[1], -2, 0, head_width=1, head_length=1, fc='orange', ec='orange',
                     linewidth=3)
            ax.text(node_pos[0] - 4, node_pos[1] + 1, f"F = {force_value} Н", fontsize=10, fontweight='bold')
        elif node_pos[0] == 110:
            ax.arrow(node_pos[0], node_pos[1], 2, 0, head_width=1, head_length=1, fc='orange', ec='orange', linewidth=3)
            ax.text(node_pos[0] + 4, node_pos[1] + 1, f"F = {force_value} Н", fontsize=10, fontweight='bold')

    ax.set_xlim(-15, 125)
    ax.set_ylim(-5, 20)
    ax.set_aspect('equal')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.show()

def plot_deformation_with_stresses(nodes, elements, U, stresses, force_nodes, force_value):
    fig, ax = plt.subplots(figsize=(14, 7))
    real_displacements = U.reshape(-1, 2)
    max_disp = np.max(np.abs(real_displacements))
    display_scale = 3 / max_disp if max_disp > 1e-10 else 1
    deformed_nodes = nodes + real_displacements * display_scale

    # Исходная сетка
    for elem in elements:
        ax.add_patch(
            patches.Polygon(nodes[elem], closed=True, fill=False, edgecolor='blue', linewidth=1, linestyle='--',
                            alpha=0.7))
        ax.add_patch(patches.Polygon(deformed_nodes[elem], closed=True, fill=False, edgecolor='red', linewidth=2))

    # Узлы
    ax.plot(nodes[:, 0], nodes[:, 1], 'bo', markersize=6, label='Исходный', alpha=0.7)
    ax.plot(deformed_nodes[:, 0], deformed_nodes[:, 1], 'rs', markersize=6, label='Деформированный')

    # Векторы деформации
    for orig, defo in zip(nodes, deformed_nodes):
        ax.arrow(orig[0], orig[1], (defo[0] - orig[0]), (defo[1] - orig[1]), head_width=0.3, head_length=0.5,
                 fc='green', ec='green', alpha=0.7)

    # Отображение сил
    for node_idx in force_nodes:
        node_pos = nodes[node_idx]
        if node_pos[0] == 0:
            ax.arrow(node_pos[0], node_pos[1], -2, 0, head_width=1, head_length=1, fc='orange', ec='orange',
                     linewidth=3)
            ax.text(node_pos[0] - 4, node_pos[1] + 1, f"F = {force_value} Н", fontsize=10, fontweight='bold')
        elif node_pos[0] == 110:
            ax.arrow(node_pos[0], node_pos[1], 2, 0, head_width=1, head_length=1, fc='orange', ec='orange', linewidth=3)
            ax.text(node_pos[0] + 4, node_pos[1] + 1, f"F = {force_value} Н", fontsize=10, fontweight='bold')

    # ВЫВОД НАПРЯЖЕНИЙ ЭЛЕМЕНТОВ
    for s in stresses:
        elem = s['index']
        centroid = np.mean([nodes[node] for node in elem], axis=0)
        val = abs(s['sigma_x'])
        # Выводим напряженность ЭЛЕМЕНТОВ
        ax.text(centroid[0], centroid[1], f"σx={s['sigma_x']:.1e}", fontsize=8, ha='center', va='center',
                bbox=dict(facecolor='yellow', alpha=0.8))

    ax.set_xlim(-15, 125)
    ax.set_ylim(-5, 20)
    ax.set_aspect('equal')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.show()

# 6. ОСНОВНОЙ КОД (ЗАПУСК)

if __name__ == "__main__":
    print("МКЭ: Профиль трубы с коррозией и трещинами (Шаг 7: Добавление второй трещины)")

    # 1. Строим сетку
    plot_mesh(nodes, elements)

    # 2. Собираем матрицу
    K_global, dof_map, local_matrices = assemble_global_stiffness(nodes, elements, E, nu, t)

    # 3. Формируем силы (растяжение с двух сторон)
    F = np.zeros(2 * len(nodes))
    for i, (x, y) in enumerate(nodes):
        if x == 0: F[2 * i] = -force
        if x == 110: F[2 * i] = force

    # 4. Закрепляем модель
    fixed_dofs = [49, 71, 58, 60]  # Закрепляем узелы по Y и по Y
    K_mod, F_mod = apply_boundary_conditions(K_global, F, fixed_dofs)

    # 5. Решаем систему
    print("Решаем систему...")
    U = np.linalg.solve(K_mod, F_mod)

    # 6. Выводим результаты (Таблица)
    print("\nПеремещения узлов (мм):")
    for i in range(len(nodes)):
        print(f"Узел {i}: Ux={U[2 * i]:.6e}, Uy={U[2 * i + 1]:.6e}")

    stresses = compute_stresses(local_matrices, U, dof_map)

    print("\nНапряжения (Па):")
    for s in stresses:
        print(f"Элемент {s['index']}: σx={s['sigma_x']:.4e}, σy={s['sigma_y']:.4e}, τxy={s['tau_xy']:.4e}")

    # 7. Визуализация
    force_nodes = [i for i, (x, y) in enumerate(nodes) if x == 0 or x == 110]
    plot_deformation(nodes, elements, U, stresses, force_nodes, force)
    plot_deformation_with_stresses(nodes, elements, U, stresses, force_nodes, force)