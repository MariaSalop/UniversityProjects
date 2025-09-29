FROM python:3.9-slim

WORKDIR /app

<<<<<<< Updated upstream
=======
RUN apt-get update && apt-get install -y ffmpeg wget curl && \
    pip install --no-cache-dir yt-dlp

>>>>>>> Stashed changes
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 80

<<<<<<< Updated upstream
CMD ["uvicorn", "FastAPI:app", "--host", "0.0.0.0", "--port", "80"]
=======
CMD ["uvicorn", "FastAPI:app", "--host", "0.0.0.0", "--port", "8000"]
>>>>>>> Stashed changes
