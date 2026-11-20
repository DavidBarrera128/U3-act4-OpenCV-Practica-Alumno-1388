import cv2
import sys

def mostrar_imagen(img, descripcion, archivo_salida=None):
    if archivo_salida:
        cv2.imwrite(archivo_salida, img)
    titulo_ventana = f"{descripcion} | NC = 1388"
    cv2.imshow(titulo_ventana, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# 1. Leer la imagen desde archivo
image = cv2.imread('avion.jpg')
if image is None:
    print("Error: No se encontró 'avion.jpg' en el directorio actual.")
    sys.exit(1)

mostrar_imagen(image, 'Imagen original')

# 2. Convertir a escala de grises
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
mostrar_imagen(gray_image, 'Escala de grises', 'gray_image.jpg')

# 3. Redimensionar la imagen a 300x200
resized_image = cv2.resize(image, (300, 200))
mostrar_imagen(resized_image, 'Redimensionada', 'resized_image.jpg')

# 4. Rotar la imagen 90 grados
(h, w) = image.shape[:2]
center = (w // 2, h // 2)
matrix = cv2.getRotationMatrix2D(center, 90, 1.0)
rotated_image = cv2.warpAffine(image, matrix, (w, h))
mostrar_imagen(rotated_image, 'Rotada 90 grados', 'rotated_image.jpg')

# 5. Aplicar desenfoque gaussiano (Gaussian Blur)
blurred_image = cv2.GaussianBlur(image, (15, 15), 0)
mostrar_imagen(blurred_image, 'Desenfoque gaussiano', 'blurred_image.jpg')

# 6. Detectar bordes usando el algoritmo Canny
edges = cv2.Canny(gray_image, 100, 200)
mostrar_imagen(edges, 'Bordes Canny', 'edges.jpg')

# 7. Dibujar un rectángulo y agregar texto
annotated_image = image.copy()
cv2.rectangle(annotated_image, (50, 50), (200, 200), (255, 0, 0), 3)
cv2.putText(annotated_image, 'OpenCV', (60, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
mostrar_imagen(annotated_image, 'Rectangulo y texto', 'output_image.jpg')

# 8. Aplicar umbralización binaria (Thresholding)
_, thresh_image = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)
mostrar_imagen(thresh_image, 'Umbralizacion binaria', 'threshold_image.jpg')

# 9. Detectar y dibujar contornos
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours_image = image.copy()
cv2.drawContours(contours_image, contours, -1, (0, 255, 0), 2)
mostrar_imagen(contours_image, 'Deteccion de contornos', 'contours_image.jpg')

print("Proceso terminado. Todas las imágenes fueron mostradas y guardadas.")
print("Act 5 OpenCV Practica Alumno David Barrera NC = 1388")