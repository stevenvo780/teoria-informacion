# Importando las bibliotecas necesarias
from multiprocessing import Process, Manager, Queue
import time
import random
import numpy as np
from collections import Counter
import heapq
from scipy.stats import entropy
import matplotlib.pyplot as plt

# Función para estimar la capacidad del canal según el teorema de Shannon
def estimate_channel_capacity(noise_level, bandwidth=1):
    # C = B * log2(1 + S/N)
    return bandwidth * np.log2(1 + 1/noise_level)

# Función para generar códigos Huffman
def generate_huffman_codes(freqs):
    # Usando una cola de prioridad para construir el árbol de Huffman
    heap = [[weight, [char, ""]] for char, weight in freqs.items()]
    heapq.heapify(heap)
    while len(heap) > 1:
        lo = heapq.heappop(heap)
        hi = heapq.heappop(heap)
        for pair in lo[1:]:
            pair[1] = '0' + pair[1]
        for pair in hi[1:]:
            pair[1] = '1' + pair[1]
        heapq.heappush(heap, [lo[0] + hi[0]] + lo[1:] + hi[1:])
    huffman_codes = sorted(heapq.heappop(
        heap)[1:], key=lambda p: (len(p[-1]), p))
    return {char: code for char, code in huffman_codes}

# Función para calcular la información mutua
def mutual_information(behavior1, behavior2, bins=10):
    # Calculando histogramas para estimar las PDFs
    hist1, _ = np.histogram(behavior1, bins=bins)
    hist2, _ = np.histogram(behavior2, bins=bins)
    joint_hist, _, _ = np.histogram2d(behavior1, behavior2, bins=bins)
    p1 = hist1 / np.sum(hist1)
    p2 = hist2 / np.sum(hist2)
    p12 = joint_hist / np.sum(joint_hist)
    # Calculando la información mutua como la suma de las entropías marginales menos la entropía conjunta
    return entropy(p1) + entropy(p2) - entropy(p12.ravel())  # type: ignore

# Función para el agente que considera todos los aspectos de la teoría de la información
def complete_agent(queue1, queue2, iterations, behavior_log, name, adapt_rate=0.1, noise_level=0.1):
    behavior = 0.5  # nivel de comportamiento inicial
    # estimación de la capacidad del canal
    capacity = estimate_channel_capacity(noise_level)
    levels = int(2 ** capacity)  # niveles de cuantización según la capacidad
    freqs = Counter(range(levels))
    huffman_codes = generate_huffman_codes(freqs)
    reverse_huffman_codes = {
        code: char for char, code in huffman_codes.items()}

    for i in range(iterations):
        # Enviar el comportamiento como un mensaje (cuantizado y codificado con Huffman)
        message = int(behavior * (levels - 1))
        coded_message = huffman_codes[message]
        # Añadir un bit de paridad para la detección de errores
        parity_bit = str(coded_message.count('1') % 2)
        coded_message += parity_bit
        # Añadir ruido (cambiando un bit aleatoriamente)
        if random.random() < noise_level:
            flip_index = random.randint(0, len(coded_message) - 1)
            coded_message = coded_message[:flip_index] + str(
                1 - int(coded_message[flip_index])) + coded_message[flip_index + 1:]
        queue1.put(coded_message)
        # Recibir mensaje y adaptar el comportamiento
        received_coded = queue2.get()
        # Verificar el bit de paridad para la detección de errores
        if received_coded.count('1') % 2 != int(received_coded[-1]):
            received_coded = received_coded[:-1]
            received = reverse_huffman_codes.get(
                received_coded, int(levels / 2)) / (levels - 1)
        else:
            received = 0.5  # default en caso de error detectado
        behavior += adapt_rate * (received - behavior)
        behavior_log.append((i, behavior, name))
        time.sleep(0.01)  # simular retardo de tiempo

# Función principal para el experimento completo
def run_complete_experiment(iterations=100, noise_level=0.1):
    manager = Manager()
    behavior_log1 = manager.list()
    behavior_log2 = manager.list()
    queue1 = Queue()
    queue2 = Queue()
    p1 = Process(target=complete_agent, args=(queue1, queue2,
                 iterations, behavior_log1, 'Agent1', 0.1, noise_level))
    p2 = Process(target=complete_agent, args=(queue2, queue1,
                 iterations, behavior_log2, 'Agent2', 0.1, noise_level))
    p1.start()
    p2.start()
    p1.join()
    p2.join()
    return list(behavior_log1), list(behavior_log2)


iterations = 100
behavior_log1, behavior_log2 = run_complete_experiment(iterations, 0.1)
time1, behavior1, _ = zip(*behavior_log1)
time2, behavior2, _ = zip(*behavior_log2)
mi = mutual_information(behavior1, behavior2)
plt.figure(figsize=(12, 6))
plt.plot(time1, behavior1, label=f'Agent 1', marker='o')
plt.plot(time2, behavior2, label=f'Agent 2', marker='x')
plt.xlabel('Time Step')
plt.ylabel('Behavior Level')
plt.title(
    f'Complete Experiment Considering All Aspects (Mutual Information: {mi:.2f})')
plt.legend()
plt.grid(True)
plt.show()
