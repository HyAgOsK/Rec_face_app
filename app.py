import cv2
import streamlit as st
import numpy as np
from deepface import DeepFace
import os
import threading
import queue
from datetime import datetime, timedelta
import pandas as pd

def analyze_and_recognize(frame, config, analyze_result, recognize_result, error_info):
    try:
        # roda análise (age, gender + spoofing se ativado)
        analyzed = DeepFace.analyze(
            img_path=frame,
            actions=['age', 'gender'],
            enforce_detection=False,
            detector_backend=config['detector_backend'],
            align=config['align'],
            silent=config['silent'],
            anti_spoofing=config['anti_spoofing'],
            expand_percentage=config['expand_percentage']
        )
        analyze_result["data"] = analyzed
        error_info["analyze_error"] = None

        # Detecta spoofing em cada face retornada
        error_info["spoofing_detected"] = False
        if config['anti_spoofing']:
            faces = analyzed if isinstance(analyzed, list) else [analyzed]
            for face in faces:
                # campo 'real_face' = False indica spoofing
                if face.get("real_face") is False:
                    error_info["spoofing_detected"] = True
                    # Marca spoofing mas NÃO levanta exceção
                    face["spoofing_detected"] = True
                else:
                    face["spoofing_detected"] = False

        # busca reconhecimento no DB
        try:
            h, w = frame.shape[:2]
            small = cv2.resize(frame, (640, int(h * (640/w))))
            recognized = DeepFace.find(
                img_path=small,
                db_path='./celebrity_faces/',
                model_name=config['model_face_recognition'],
                enforce_detection=False,
                detector_backend=config['detector_backend'],
                distance_metric=config['distance_metric'],
                align=config['align'],
                silent=config['silent']
            )
            recognize_result["data"] = recognized
        except Exception as rec_error:
            recognize_result["data"] = None

    except Exception as e:
        error_msg = str(e)
        
        # CORREÇÃO PRINCIPAL: Se o erro é de spoofing, não bloqueia o processamento
        if "spoof" in error_msg.lower() or "anti" in error_msg.lower():
            # Cria resultado artificial para mostrar spoofing
            try:
                # Tenta detectar faces básicas para localizar onde desenhar
                face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, 1.1, 4)
                
                if len(faces) > 0:
                    # Usa a primeira face detectada
                    (x, y, w, h) = faces[0]
                    fake_result = [{
                        'region': {'x': x, 'y': y, 'w': w, 'h': h},
                        'age': 25,  # Idade padrão
                        'dominant_gender': 'Man',  # Gênero padrão
                        'real_face': False,
                        'spoofing_detected': True
                    }]
                else:
                    # Se não detectar faces, usa área central
                    h, w = frame.shape[:2]
                    fake_result = [{
                        'region': {'x': w//4, 'y': h//4, 'w': w//2, 'h': h//2},
                        'age': 25,
                        'dominant_gender': 'Man',
                        'real_face': False,
                        'spoofing_detected': True
                    }]
                
                analyze_result["data"] = fake_result
                error_info["spoofing_detected"] = True
                error_info["analyze_error"] = None  # Não mostra erro
                recognize_result["data"] = None
                
            except Exception as detect_error:
                # Resultado padrão se tudo falhar
                h, w = frame.shape[:2]
                analyze_result["data"] = [{
                    'region': {'x': 50, 'y': 50, 'w': w-100, 'h': h-100},
                    'age': 25,
                    'dominant_gender': 'Man',
                    'real_face': False,
                    'spoofing_detected': True
                }]
                error_info["spoofing_detected"] = True
                error_info["analyze_error"] = None
                recognize_result["data"] = None
        else:
            # Erro normal - bloqueia processamento
            analyze_result["data"] = None
            recognize_result["data"] = None
            error_info["analyze_error"] = str(e)
            error_info["spoofing_detected"] = False


def facesentiment(config, error_info):
    cap = cv2.VideoCapture(0)
    stframe = st.image([])
    log_placeholder = st.empty()

    analyze_result, recognize_result = {}, {}
    recognized_log = []

    frame_queue = queue.Queue(maxsize=1)
    stop_event = threading.Event()

    def worker():
        while not stop_event.is_set():
            frame = frame_queue.get()
            if frame is None:
                break
            analyze_and_recognize(frame, config, analyze_result, recognize_result, error_info)

    threading.Thread(target=worker, daemon=True).start()

    frame_counter = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            st.warning("⚠️ Webcam não detectada.")
            break

        # dispara worker a cada 3 frames
        if frame_counter % 3 == 0 and frame_queue.empty():
            frame_queue.put(frame.copy())

        # Processar resultados
        frame_overlay = frame.copy()
        result = analyze_result.get("data")
        
        if result:
            for face in result:
                region = face.get('region')
                if not region:
                    continue

                x, y, w, h = region['x'], region['y'], region['w'], region['h']
                idade = face.get('age', 0)
                genero = 'Homem' if face.get('dominant_gender') == 'Man' else 'Mulher'

                # Verifica spoofing
                is_spoofing = face.get("spoofing_detected", False)
                if config['anti_spoofing'] and face.get("real_face") is False:
                    is_spoofing = True


                if is_spoofing:
                    name = "SPOOFING DETECTADO"
                    color = (0, 0, 255)  # Vermelho
                    erro_status = "Spoofing"
                else:
                    name = "Desconhecido"
                    color = (0, 255, 0)  # Verde
                    erro_status = ""
                    
                    # Tenta reconhecer
                    db_result = recognize_result.get("data")
                    if db_result and len(db_result) > 0 and not db_result[0].empty:
                        best_match = db_result[0].iloc[0]
                        identity = best_match['identity'].split('/')[-1].split('.')[0]
                        name = identity.split('\\')[-2] if '\\' in identity else identity

                # Log
                now = datetime.now()
                dois_min = now - timedelta(minutes=2)
                duplicate = any(r["Nome"]==name and r["Datetime"]>=dois_min for r in recognized_log)
                
                if (is_spoofing or name not in ["Desconhecido"]) and not duplicate:
                    recognized_log.append({
                        "Nome": name,
                        "Idade": int(idade),
                        "Gênero": genero,
                        "Hora": now.strftime("%H:%M:%S"),
                        "Erro": erro_status,
                        "Datetime": now
                    })
                    df = pd.DataFrame(recognized_log[::-1])
                    df_display = df.drop(columns=["Datetime"])
                    log_placeholder.dataframe(df_display)
                    df_display.to_csv("log_reconhecimento.csv", index=False)

                # Desenha no rosto
                display_text = name
                
                # Caixa de texto maior e mais visível
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.8
                thickness = 2
                (tw, th), baseline = cv2.getTextSize(display_text, font, font_scale, thickness)
                
                # Retângulo da face
                cv2.rectangle(frame_overlay, (x, y), (x + w, y + h), color, 3)
                
                # Fundo do texto
                cv2.rectangle(frame_overlay, (x, y - th - 15), (x + tw + 10, y), color, -1)
                
                # Texto principal
                cv2.putText(frame_overlay, display_text, (x + 5, y - 8), 
                           font, font_scale, (255, 255, 255), thickness)
                
                # Informação adicional
                if config['anti_spoofing']:
                    status_text = f"Real: {'Não' if is_spoofing else 'Sim'}"
                    cv2.putText(frame_overlay, status_text, (x + 5, y + h + 25),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        frame_rgb = cv2.cvtColor(frame_overlay, cv2.COLOR_BGR2RGB)
        stframe.image(frame_rgb, channels="RGB")
        frame_counter += 1

    stop_event.set()
    cap.release()


def main():
    st.set_page_config(page_title="Reconhecimento Facial em Tempo Real", layout="centered")
    st.sidebar.title("Configurações")

    config = {
        "detector_backend": st.sidebar.selectbox("Detector",
            ['retinaface','opencv','mtcnn','ssd','dlib','mediapipe','yolov8','centerface','skip']),
        "align": st.sidebar.checkbox("Alinhar rostos", value=True),
        "anti_spoofing": st.sidebar.checkbox("Anti-spoofing", value=False),
        "distance_metric": st.sidebar.selectbox("Métrica de distância",
            ['cosine','euclidean','euclidean_l2']),
        "model_face_recognition": st.sidebar.selectbox("Modelo de reconhecimento",
            ["Facenet512","VGG-Face","Facenet","OpenFace","DeepFace",
             "DeepID","Dlib","ArcFace","SFace","GhostFaceNet"]),
        "expand_percentage": st.sidebar.slider("Expandir detecção (%)", 0,100,0),
        "silent": st.sidebar.checkbox("Modo silencioso", value=True)
    }

    st.title("🧠 Aplicativo de Reconhecimento Facial")
    
    if config['anti_spoofing']:
        st.info("🛡️ Anti-spoofing ativado! Tentativas de spoofing serão detectadas.")
    
    st.subheader("📸 Cadastre um novo usuário")
    uploaded = st.file_uploader("Selecione uma imagem", type=["jpg","jpeg","png"])

    error_info = {"analyze_error": None, "spoofing_detected": False}

    if uploaded is not None:
        folder = ''.join(filter(str.isalpha, uploaded.name.split('.')[0]))
        path = f"./celebrity_faces/{folder}/"
        os.makedirs(path, exist_ok=True)
        with open(f"{path}/{uploaded.name}", "wb") as f:
            f.write(uploaded.getbuffer())
        st.success(f"Imagem {uploaded.name} adicionada com sucesso!")

    facesentiment(config, error_info)


if __name__ == "__main__":
    main()