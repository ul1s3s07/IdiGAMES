import cv2
import numpy as np
import mediapipe as mp

# Inicialización de MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Configuración de la cámara
cap = cv2.VideoCapture(0)
drawing_canvas = None  # El lienzo se creará dinámicamente según el tamaño de la cámara

# Variables para el dibujo
drawing = False
prev_x, prev_y = None, None  # Posiciones previas del dedo índice
brush_color = (255, 0, 0)  # Color inicial (rojo)

# Colores disponibles y sus posiciones en la barra de selección
colors = {
    (0, 0, 255): (50, 50),  # Rojo
    (0, 255, 0): (150, 50),  # Verde
    (255, 0, 0): (250, 50),  # Azul
    (0, 255, 255): (350, 50),  # Amarillo
    (255, 255, 255): (450, 50),  # Blanco
    (0, 0, 0): (550, 50),  # Negro
}

# Funciones para determinar el estado de la mano
def mano_esta_cerrada(hand_landmarks):
    """Determina si la mano está cerrada (puño)."""
    dedos_cerrados = [
        hand_landmarks.landmark[8].y > hand_landmarks.landmark[6].y,  # Índice
        hand_landmarks.landmark[12].y > hand_landmarks.landmark[10].y,  # Medio
        hand_landmarks.landmark[16].y > hand_landmarks.landmark[14].y,  # Anular
        hand_landmarks.landmark[20].y > hand_landmarks.landmark[18].y  # Meñique
    ]
    return all(dedos_cerrados)

def palma_esta_abierta(hand_landmarks):
    """Determina si la palma está abierta (todos los dedos extendidos)."""
    dedos_abiertos = [
        hand_landmarks.landmark[8].y < hand_landmarks.landmark[6].y,  # Índice
        hand_landmarks.landmark[12].y < hand_landmarks.landmark[10].y,  # Medio
        hand_landmarks.landmark[16].y < hand_landmarks.landmark[14].y,  # Anular
        hand_landmarks.landmark[20].y < hand_landmarks.landmark[18].y  # Meñique
    ]
    return all(dedos_abiertos)

# Ciclo principal
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Obtener las dimensiones del fotograma
    frame_height, frame_width, _ = frame.shape

    # Crear el lienzo si aún no se ha creado o si las dimensiones cambian
    if drawing_canvas is None or drawing_canvas.shape[:2] != frame.shape[:2]:
        drawing_canvas = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)

    # Preparar la imagen para Mediapipe
    frame = cv2.flip(frame, 1)  # Espejo
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    # Dibujar barra de selección de colores
    for color, (x, y) in colors.items():
        cv2.circle(frame, (x, y), 20, color, -1)
        if color == brush_color:
            cv2.circle(frame, (x, y), 25, (255, 255, 255), 2)  # Resaltar el color seleccionado

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Coordenadas del dedo índice
            index_finger_tip = hand_landmarks.landmark[8]
            index_x = int(index_finger_tip.x * frame_width)
            index_y = int(index_finger_tip.y * frame_height)

            # Verificar el estado de la mano
            if mano_esta_cerrada(hand_landmarks):
                drawing = False  # Detener el dibujo
            elif palma_esta_abierta(hand_landmarks):
                drawing = False  # Detener el dibujo
                # Borrar el área tocada por la palma
                palm_center_x = int(hand_landmarks.landmark[9].x * frame_width)
                palm_center_y = int(hand_landmarks.landmark[9].y * frame_height)
                cv2.circle(drawing_canvas, (palm_center_x, palm_center_y), 50, (0, 0, 0), -1)
            else:
                drawing = True  # Permitir el dibujo

            # Verificar si el dedo índice selecciona un color
            for color, (x, y) in colors.items():
                if (x - 20) < index_x < (x + 20) and (y - 20) < index_y < (y + 20):
                    brush_color = color

            # Dibujar si el dedo índice está activo y en movimiento
            if drawing:
                if prev_x is not None and prev_y is not None:
                    cv2.line(drawing_canvas, (prev_x, prev_y), (index_x, index_y), brush_color, 5)
                prev_x, prev_y = index_x, index_y
            else:
                prev_x, prev_y = None, None  # Resetear la posición previa cuando no se dibuja

    # Combinar el lienzo con el marco de la cámara
    combined_frame = cv2.addWeighted(frame, 0.5, drawing_canvas, 0.5, 0)

    # Mostrar la salida
    cv2.imshow('DibujAR', combined_frame)

    # Salir con la tecla ESC
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Liberar recursos
cap.release()
cv2.destroyAllWindows()
