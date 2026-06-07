import cv2
import numpy as np
import time

# Cargar las clases de COCO
with open('coco.names', 'r') as f:
    classes = [line.strip() for line in f.readlines()]

# Generar colores aleatorios para cada clase
COLORS = np.random.uniform(0, 255, size=(len(classes), 3))

# Cargar el modelo YOLOv3 entrenado con MS COCO
net = cv2.dnn.readNet('yolov3.weights', 'yolov3.cfg')

# Configurar backend y target (opcional, para usar GPU si está disponible)
# net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
# net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

# Obtener nombres de las capas de salida
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Definir tamaño de salida
OUTPUT_WIDTH = 1280
OUTPUT_HEIGHT = 720

# Cambiar a True para usar video, False para usar imagen
USE_VIDEO = False
VIDEO_PATH = 'img/shib.mp4'  # Ruta del video
IMAGE_PATH = 'img/shibuya.jpg'  # Ruta de la imagen

def detect_objects(frame):
    """Función para detectar objetos en un frame"""
    height, width, channels = frame.shape
    
    # Crear blob desde el frame
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
    
    # Pasar el blob a través de la red
    net.setInput(blob)
    outputs = net.forward(output_layers)
    
    # Procesar las detecciones
    boxes = []
    confidences = []
    class_ids = []
    
    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            
            # Filtrar detecciones débiles
            if confidence > 0.5:
                # Obtener coordenadas del bounding box
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                
                # Coordenadas de la esquina superior izquierda
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    
    # Aplicar Non-Maximum Suppression para eliminar detecciones duplicadas
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    
    # Dibujar las detecciones
    font = cv2.FONT_HERSHEY_SIMPLEX
    if len(indexes) > 0:
        for i in indexes.flatten():
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = confidences[i]
            color = COLORS[class_ids[i]]
            
            # Dibujar rectángulo y etiqueta
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            text = f"{label}: {confidence:.2f}"
            cv2.putText(frame, text, (x, y - 5), font, 0.5, color, 2)
    
    return frame

if USE_VIDEO:
    # Procesar video
    cap = cv2.VideoCapture(VIDEO_PATH)
    
    if not cap.isOpened():
        print(f"Error: No se pudo abrir el video '{VIDEO_PATH}'")
        exit()
    
    # Obtener propiedades del video
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video: {fps} FPS, {frame_count} frames")
    
    # Configurar video de salida
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('yolo_detection_video.mp4', fourcc, fps, (OUTPUT_WIDTH, OUTPUT_HEIGHT))
    
    frame_num = 0
    while True:
        ret, frame = cap.read()
        
        if not ret:
            break
        
        # Escalar el frame
        frame = cv2.resize(frame, (OUTPUT_WIDTH, OUTPUT_HEIGHT), interpolation=cv2.INTER_LINEAR)
        
        # Detectar objetos
        start_time = time.time()
        frame = detect_objects(frame)
        end_time = time.time()
        
        # Calcular FPS
        processing_fps = 1 / (end_time - start_time)
        cv2.putText(frame, f"FPS: {processing_fps:.1f}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Mostrar frame
        cv2.imshow('YOLOv3 Video Detection', frame)
        
        # Guardar frame en video de salida
        out.write(frame)
        
        frame_num += 1
        print(f"Procesando frame {frame_num}/{frame_count}", end='\r')
        
        # Presionar 'q' para salir
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    out.release()
    print(f"\n¡Video guardado como 'yolo_detection_video.mp4'!")
    
else:
    # Procesar imagen
    image = cv2.imread(IMAGE_PATH)
    
    if image is None:
        print(f"Error: No se pudo cargar la imagen '{IMAGE_PATH}'")
        exit()
    
    # Escalar la imagen
    image = cv2.resize(image, (OUTPUT_WIDTH, OUTPUT_HEIGHT), interpolation=cv2.INTER_LINEAR)
    print(f"Imagen escalada a: {OUTPUT_WIDTH}x{OUTPUT_HEIGHT}")
    
    # Detectar objetos
    image = detect_objects(image)
    
    # Mostrar y guardar resultado
    cv2.imshow('YOLOv3 Object Detection', image)
    cv2.imwrite('yolo_detection_result.jpg', image)
    cv2.waitKey(0)

cv2.destroyAllWindows()

