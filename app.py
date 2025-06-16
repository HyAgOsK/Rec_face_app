import cv2
import streamlit as st
import numpy as np
from deepface import DeepFace
import os

# Função para comparar com base de dados
def stream_frame(frame,model_face_recognition, detector_backend, distance_metric, enforce_detection, align, silent):
    result = DeepFace.find(
        img_path=frame,
        db_path='./celebrity_faces/',
        model_name=model_face_recognition,
        detector_backend=detector_backend,
        distance_metric=distance_metric,
        enforce_detection=enforce_detection,
        align=align,
        silent=silent
    )
    return result

# Função para análise facial (idade, emoção, etc.)
def analyze_frame(frame, detector_backend, enforce_detection, align, anti_spoofing, silent, expand_percentage):
    result = DeepFace.analyze(
        img_path=frame,
        actions=['age', 'gender', 'race', 'emotion'],
        enforce_detection=enforce_detection,
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
    
    box_height = 140
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], box_height), (255, 255, 255), -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    y_offset = 20
    for text in texts:
        cv2.putText(frame, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        y_offset += 20
    return frame

# Função principal de análise de rosto
def facesentiment(choice, 
                  detector_backend,
                  enforce_detection,
                  align,
                  anti_spoofing,
                  distance_metric,
                  model_face_recognition,
                  expand_percentage,
                  silent):

    cap = cv2.VideoCapture(0)
    stframe = st.image([])

    while True:
        ret, frame = cap.read()
        if not ret:
            st.warning("⚠️ Não foi possível capturar imagem da webcam.")
            continue

        try:
            result = analyze_frame(frame, detector_backend, enforce_detection, align, anti_spoofing, silent, expand_percentage)
        except Exception as e:
            st.error(f"Erro na análise de rosto: {e}")
            continue

        # Desenhar caixa ao redor do rosto
        try:
            region = result[0]["region"]
            x, y, w, h = region["x"], region["y"], region["w"], region["h"]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        except Exception:
            pass  # Caso não encontre face

        # Coleta de dados para exibição
        texts = [
            f"Age: {result[0]['age']}",
            f"Confidence: {round((result[0]['face_confidence'])*100, 2)}",
            f"Gender: {result[0]['dominant_gender']} ({round(result[0]['gender'][result[0]['dominant_gender']], 2)})",
            f"Race: {result[0]['dominant_race']}",
            f"Emotion: {result[0]['dominant_emotion']} ({round(result[0]['emotion'][result[0]['dominant_emotion']], 2)})",
        ]

        # Reconhecimento facial se for essa a escolha
        if choice == "Face recognition":
            try:
                matches = stream_frame(frame,model_face_recognition, detector_backend, distance_metric, enforce_detection, align, silent)
                if matches and len(matches) > 0 and not matches[0].empty:
                    best_match = matches[0].iloc[0]
                    identity = best_match['identity'].split('/')[-1]
                    identity = identity.split('.')[0] 
                    name = identity.split('\\')[-2]
                    if result[0]['face_confidence'] < 0.90:
                        texts.append(f"Face: Not Recognized")
                    else:
                        texts.append(f"Recognized as: {name}")
                    
                else:
                    texts.append("Face: Not Recognized.")
            except Exception as e:
                texts.append(f"Recognition error: {e}")

        # Processar imagem para exibição
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_overlay = overlay_text_on_frame(frame_rgb, texts)
        stframe.image(frame_overlay, channels="RGB")

# Interface Streamlit
def main():
    st.set_page_config(page_title="Real-Time Face Recognition", layout="centered")

    st.sidebar.title("Settings")
    activities = ["Face analyse", "Face recognition"]
    choice = st.sidebar.selectbox("Select Activity", activities)


    #------------------------------------
    # Features for face analysis and recognition
    detector_backend = st.sidebar.selectbox(
        "Select the backend for face detector_backend",
        options=['opencv', 'retinaface','mtcnn', 'ssd', 'dlib', 'mediapipe', 'yolov8', 'centerface' , 'skip'],
    )
    enforce_detection = st.sidebar.checkbox(
        "Enforce detection",
        value=False,
        help="If no face is detected in an image, raise an exception."
    )
    align = st.sidebar.checkbox(
        "Align faces",
        value=True,
        help="Perform alignment based on the eye positions."
    )
    anti_spoofing = st.sidebar.checkbox(
        "Anti-spoofing",
        value=False,
        help="Enable anti-spoofing to detect fake faces."
    )
    distance_metric = st.sidebar.selectbox(
        "Select the distance metric for similarity",
        options=['cosine', 'euclidean', 'euclidean_l2'],
    )
    model_face_recognition = st.sidebar.selectbox(
        "Select the model for face recognition",
        options=["VGG-Face", "Facenet", "Facenet512", "OpenFace", "DeepFace", "DeepID", "Dlib", "ArcFace", "SFace", "GhostFaceNet"]
    )
    expand_percentage = st.sidebar.slider(
        "Expand percentage for face detection",
        min_value=0, max_value=100, value=0,
        help="Expand detected facial area with a percentage."
    )
    silent = st.sidebar.checkbox(
        "Silent mode",
        value=True,
        help="Suppress or allow some log messages for a quieter analysis process."
    )
    
    #-------------------------------------
    
    st.title("Face recognition APP", anchor="center")

    # escolher arquivo de imagem para subir no banco de dados
    uploaded_file = st.file_uploader("Upload an image to add to the database", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        upload_name_dir = uploaded_file.name.split('.')[0]
        upload_name_dir = ''.join(filter(str.isalpha, upload_name_dir))
        if not os.path.exists(f"./celebrity_faces/{upload_name_dir}"):
            os.makedirs(f"./celebrity_faces/{upload_name_dir}")
        with open(f"./celebrity_faces/{upload_name_dir}/{uploaded_file.name}", "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success(f"Image {uploaded_file.name} added to the database.")
        
    facesentiment(choice, 
                  detector_backend=detector_backend,
                  enforce_detection=enforce_detection,
                  align=align,
                  anti_spoofing=anti_spoofing,
                  distance_metric=distance_metric,
                  model_face_recognition=model_face_recognition,
                  expand_percentage=expand_percentage,
                  silent=silent)
   
    
                
if __name__ == "__main__": 
    main()
