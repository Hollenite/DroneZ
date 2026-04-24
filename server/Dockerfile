FROM python:3.12-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PORT=7860

COPY server/requirements.txt /tmp/server-requirements.txt
RUN pip install --no-cache-dir -r /tmp/server-requirements.txt

COPY . /app
RUN pip install --no-cache-dir -e .

EXPOSE 7860

CMD ["sh", "-c", "uvicorn server.app:app --host 0.0.0.0 --port ${PORT}"]
