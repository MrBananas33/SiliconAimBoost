import torch
import torch.nn.functional as F
import cv2
from ultralytics import YOLO 
import numpy as np

# Verifica si MPS está disponible
mps_disponible = torch.backends.mps.is_available()

# Imprime si MPS está disponible
print(f"MPS disponible: {mps_disponible}")

# Si MPS está disponible, asigna el dispositivo a MPS, de lo contrario a CPU
device = torch.device("mps" if mps_disponible else "cpu")

# Imprime el dispositivo actual
print(f"Dispositivo actual: {device}")

# Verifica si MPS está disponible y asigna el dispositivo
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

# Carga el modelo YOLO y lo asigna al dispositivo MPS
model = YOLO("yolov5nu.pt").to(device)

# Abre el video
cap = cv2.VideoCapture("people.mp4")

# Bucle para leer continuamente los fotogramas del video
while True:
    ret, frame = cap.read()

    # Si no se puede leer un fotograma, rompe el bucle
    if not ret:
        break

    # Convierte el fotograma a un tensor, lo normaliza y lo mueve al dispositivo MPS
    frame_tensor = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
    frame_tensor = frame_tensor.unsqueeze(0).to(device)

    # Redimensiona el tensor para que tenga una altura y anchura divisibles por 32
    stride = 32
    height, width = frame_tensor.shape[2], frame_tensor.shape[3]
    new_height = (height // stride) * stride
    new_width = (width // stride) * stride
    frame_tensor = F.interpolate(frame_tensor, size=(new_height, new_width), mode='bilinear', align_corners=False)

    # Realiza la detección con el modelo YOLO
    results = model(frame_tensor)
    result = results[0]
    bboxes = np.array(result.boxes.xyxy.cpu(), dtype=int)
    print(bboxes)
    for bbox in bboxes:
        (x, y, x2, y2) = bbox
        cv2.rectangle(frame, (x, y), (x2, y2), (0, 0, 225), 2)

    # Muestra los resultados
    print(results)
    
    # Muestra el fotograma en una ventana
    cv2.imshow("Img", frame)

    # Espera a que se presione una tecla
    key = cv2.waitKey(27)

    # Si se presiona 'q', rompe el bucle
    if key == ord('q'):
        break

# Libera el objeto de captura y cierra todas las ventanas
cap.release()
cv2.destroyAllWindows()
