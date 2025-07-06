import cv2
import streamlit as st
import numpy as np
from deepface import DeepFace
import os
import threading
import queue
import time

# Função para comparar com base de dados
def stream_frame(frame, model_face_recognition, detector_backend, distance_metric, align, silent):
    result = DeepFace.find(
        img_path=frame,
        db_path='./celebrity_faces/',
        model_name=model_face_recognition,
        enforce_detection=False,
        detector_backend=detector_backend,
        distance_metric=distance_metric,
        align=align,
        silent=silent
    )
    return result

# Função para análise facial (idade, emoção, etc.)
def analyze_frame(frame, detector_backend, align, anti_spoofing, silent, expand_percentage):
    result = DeepFace.analyze(
        img_path=frame,
        actions=['age', 'gender', 'race', 'emotion'],
        enforce_detection=False,
        detector_backend=detector_backend,
        align=align,
        silent=silent,
        anti_spoofing=anti_spoofing,
        expand_percentage=expand_percentage
    )
    return result

# Sobreposição de texto no frame
def overlay_text_on_frame(frame, texts):
    overlay = frame.copy()
    alpha = 0.9
    box_height = 70
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], box_height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    y_offset = 20
    for text in texts:
        cv2.putText(frame, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y_offset += 20
    return frame


# Função principal de análise de rosto
def facesentiment(detector_backend,
                  align,
                  anti_spoofing,
                  distance_metric,
                  model_face_recognition,
                  expand_percentage,
                  silent):

    cap = cv2.VideoCapture('Luciano.mp4')
    stframe = st.image([])

    analyze_result = {}
    recognition_result = {}
    last_frame = None
    error_info = {"analyze_error": None}  # Para armazenar erros de forma segura
    frame_queue = queue.Queue(maxsize=2)

    # Thread para análise facial
    def analyze_worker():
        while True:
            frame = frame_queue.get()
            if frame is None:
                break
            try:
                result = analyze_frame(frame, detector_backend, align, anti_spoofing, silent, expand_percentage)
                analyze_result["data"] = result
                error_info["analyze_error"] = None
            except Exception as e:
                analyze_result["data"] = None
                error_info["analyze_error"] = str(e)

    # Thread para reconhecimento facial
    def recognize_worker():
        while True:
            frame = frame_queue.get()
            if frame is None:
                break
            try:
                result = stream_frame(frame, model_face_recognition, detector_backend, distance_metric, align, silent)
                recognition_result["data"] = result
            except Exception as e:
                recognition_result["data"] = None
                print("Reconhecimento facial falhou:", e)

    # Inicia threads
    threading.Thread(target=analyze_worker, daemon=True).start()
    threading.Thread(target=recognize_worker, daemon=True).start()

    while True:
        ret, frame = cap.read()
        if not ret:
            st.warning("⚠️ Webcam não detectada.")
            break

        last_frame = frame.copy()

        # Coloca na fila se tiver espaço
        try:
            if frame_queue.qsize() < 2:
                frame_queue.put_nowait(frame.copy())
        except queue.Full:
            pass

        texts = []
        result = analyze_result.get("data", None)
        matches = recognition_result.get("data", None)

        # Verifica erro de spoofing
        if error_info["analyze_error"]:
            if "spoof" in error_info["analyze_error"].lower():
                texts.append("Imagem de fotos, fraude!")
            else:
                texts.append(f"Erro: {error_info['analyze_error']}")

        elif result:
            gender = 'Homem' if result[0]['dominant_gender'] == 'Man' else 'Mulher'

            if matches and len(matches) > 0 and not matches[0].empty:
                try:
                    best_match = matches[0].iloc[0]
                    identity = best_match['identity'].split('/')[-1].split('.')[0]
                    name = identity.split('\\')[-2]
                    if result and result[0]['face_confidence'] < 0.90:
                        texts.append("Rosto nao encontrado")
                    else:
                        texts.append(f"Recognized as: {name}")
                        texts.append(f"Age: {result[0]['age']}")
                        texts.append(f"Gender: {gender}")
                except Exception as e:
                    texts.append(f"Erro no reconhecimento: {e}")
            else:
                texts.append("Face: Not Recognized.")

        frame_overlay = last_frame.copy()

        if result:
            for face in result:
                region = face.get('region', None)
                if region:
                    x, y, w, h = region['x'], region['y'], region['w'], region['h']
                    face_crop = frame[y:y+h, x:x+w]

                    name = "Desconhecido"
                    try:
                        single_match = DeepFace.find(
                            img_path=face_crop,
                            db_path='./celebrity_faces/',
                            model_name=model_face_recognition,
                            enforce_detection=False,
                            detector_backend=detector_backend,
                            distance_metric=distance_metric,
                            align=align,
                            silent=True
                        )

                        if single_match and len(single_match) > 0 and not single_match[0].empty:
                            best_match = single_match[0].iloc[0]
                            identity = best_match['identity'].split('/')[-1].split('.')[0]
                            name = identity.split('\\')[-2]
                    except:
                        pass

                    # Desenha o retângulo e nome
                    cv2.rectangle(frame_overlay, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.rectangle(frame_overlay, (x, y - 20), (x + len(name)*10, y), (0, 255, 0), -1)
                    cv2.putText(frame_overlay, name, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        else:
            pass


        # Adiciona texto fixo no topo (ex: idade, emoção, erros)
        frame_overlay = overlay_text_on_frame(frame_overlay, texts)

        frame_rgb = cv2.cvtColor(frame_overlay, cv2.COLOR_BGR2RGB)
        stframe.image(frame_rgb, channels="RGB", use_container_width=True)


        time.sleep(0.03)  # Pausa pequena para não sobrecarregar

# Interface Streamlit
def main():
    st.set_page_config(page_title="Real-Time Face Recognition", layout="centered")
    st.sidebar.title("Settings")

    detector_backend = st.sidebar.selectbox(
        "Select the backend for face detector_backend",
        options=['opencv', 'retinaface', 'mtcnn', 'ssd', 'dlib', 'mediapipe', 'yolov8', 'centerface', 'skip'],
    )
    align = st.sidebar.checkbox("Align faces", value=True)
    anti_spoofing = st.sidebar.checkbox("Anti-spoofing", value=False)
    distance_metric = st.sidebar.selectbox(
        "Select the distance metric for similarity",
        options=['cosine', 'euclidean', 'euclidean_l2'],
    )
    model_face_recognition = st.sidebar.selectbox(
        "Select the model for face recognition",
        options=["Facenet512", "VGG-Face", "Facenet", "OpenFace", "DeepFace", "DeepID", "Dlib", "ArcFace", "SFace", "GhostFaceNet"]
    )
    expand_percentage = st.sidebar.slider(
        "Expand percentage for face detection",
        min_value=0, max_value=100, value=0
    )
    silent = st.sidebar.checkbox("Silent mode", value=True)

    st.title("Face recognition APP", anchor="center")

    st.subheader("Cadastre o usuário abaixo")
    uploaded_file = st.file_uploader("Clique abaixo", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        upload_name_dir = uploaded_file.name.split('.')[0]
        upload_name_dir = ''.join(filter(str.isalpha, upload_name_dir))
        if not os.path.exists(f"./celebrity_faces/{upload_name_dir}"):
            os.makedirs(f"./celebrity_faces/{upload_name_dir}")
        with open(f"./celebrity_faces/{upload_name_dir}/{uploaded_file.name}", "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success(f"Image {uploaded_file.name} added to the database.")

    facesentiment(detector_backend=detector_backend,
                  align=align,
                  anti_spoofing=anti_spoofing,
                  distance_metric=distance_metric,
                  model_face_recognition=model_face_recognition,
                  expand_percentage=expand_percentage,
                  silent=silent)

if __name__ == "__main__":
    main()
