# AutoVisionAR: Presentación de Coches con Realidad Aumentada

AutoVisionAR es una aplicación web basada en Flask que utiliza realidad aumentada para mostrar coches. Detecta marcadores ArUco a través de una cámara y superpone modelos de coches e información sobre los marcadores.

## Características

- Detección en tiempo real de marcadores ArUco
- Superposición de modelos de coches en los marcadores con realidad aumentada
- Funcionalidad de comandos de voz para alternar la visualización de modelos y precios
- Seguimiento de coches detectados
- Guía de usuario

## Requisitos

- Python 3.7+
- OpenCV
- Flask
- NumPy
- SpeechRecognition

### Instalación de dependencias
Para la gestión de dependencias he decidido usar Poetry. Poetry simplifica la gestión de dependencias y entornos virtuales en proyectos con Python, asegurando consistencia y facilitando la gestión de dependencias.


Para instalar Poetry
```python
$ pip install poetry
```

Para insalar las dependencias:
```python
poetry install
```

Alternativamente, también se proporciona un archivo requirements.txt por si el usuario no quiere utilizar Poetry.

## Uso

1. Ejecuta app.py
2. Abre un navegador web y navega a `http://localhost:5000`.

3. Usa el menú de navegación para acceder a las diferentes características:
- Inicio: Página principal
- Manual: Guía de usuario
- Cámara: Vista de realidad aumentada
- Coches Detectados: Lista de coches detectados

4. En la vista de Cámara, apunta la cámara a los marcadores ArUco para ver los modelos de coches y la información.

5. Usa comandos de voz para controlar la visualización:
- "modelo": Muestra el modelo del coche en pantalla
- "precio": Muestra el precio del coche en pantalla

## Estructura del Proyecto

- `app.py`: Aplicación principal de Flask
- `utils.py`: Funciones del programa
- `config.py`: Configuración de la aplicación
- `data/car_data.csv`: Base de datos de información de coches
- `static/images/`: Imágenes de modelos de coches
- `templates/`: Plantillas HTML para las páginas web
