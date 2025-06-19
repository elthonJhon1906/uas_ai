from collections import deque

def bfs_shortest_path_with_distance(graph, start, goal):
    queue = deque([([start], 0)])  # Menyimpan tuple: (jalur, total_jarak)
    visited = {start}

    while queue:
        path, total_distance = queue.popleft()
        current_node = path[-1]

        if current_node == goal:
            return path, total_distance

        for neighbor, distance in graph.get(current_node, {}).items():
            if neighbor not in visited:
                visited.add(neighbor)
                new_path = path + [neighbor]
                new_total_distance = total_distance + distance
                queue.append((new_path, new_total_distance))

    return None, None

# Graf dengan jarak antar provinsi (dalam km - perkiraan)
graph_sumatera = {
    'Aceh': {'Sumatera Utara': 440},
    'Sumatera Utara': {'Riau': 580},
    'Riau': {'Jambi': 280, 'Sumatera Barat': 300},
    'Sumatera Barat': {'Jambi': 250},
    'Jambi': {'Sumatera Selatan': 340, 'Bengkulu': 370},
    'Bengkulu': {'Sumatera Selatan': 220},
    'Sumatera Selatan': {'Lampung': 290}
}

start_province = 'Aceh'
goal_province = 'Lampung'

print(f"Mencari jalur dari {start_province} ke {goal_province} menggunakan BFS...\n")
shortest_route, total_distance = bfs_shortest_path_with_distance(graph_sumatera, start_province, goal_province)

if shortest_route:
    print("Jalur terpendek ditemukan:")
    print(" -> ".join(shortest_route))
    print(f"Total jarak: {total_distance} km")
else:
    print(f"Tidak dapat menemukan jalur dari {start_province} ke {goal_province} dengan graf yang diberikan.")
