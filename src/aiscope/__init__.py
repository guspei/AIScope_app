import os

# Ultralytics instala paquetes por su cuenta cuando le falta algo; eso rompe el entorno fijado.
os.environ.setdefault("YOLO_AUTOINSTALL", "false")
