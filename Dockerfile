FROM python:3.11-slim
LABEL "language"="python"
LABEL "framework"="streamlit"
WORKDIR /src
COPY . .
RUN pip install --no-cache-dir -r requirements.txt
EXPOSE 8501
CMD ["streamlit", "run", "paired_app.py", "--server.port=8501", "--server.address=0.0.0.0"]