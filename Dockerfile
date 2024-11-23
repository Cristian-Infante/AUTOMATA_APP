# Usa una imagen base oficial de Python
FROM python:3.10-slim

# Establece el directorio de trabajo
WORKDIR /app

# Instala las dependencias del sistema (incluye Graphviz)
RUN apt-get update && apt-get install -y \
    graphviz \
    && rm -rf /var/lib/apt/lists/*

# Instala las dependencias de Python
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copia el código fuente de tu aplicación
COPY . .

# Expone el puerto para Railway
EXPOSE 8000

# Comando de inicio
CMD ["python", "manage.py", "runserver", "0.0.0.0:8000"]
