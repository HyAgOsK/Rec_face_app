import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode
import av
import cv2
from deepface import DeepFace
import os
import threading
from datetime import datetime, timedelta
import pandas as pd
import logging

# Configurar logging para reduzir warnings
logging.getLogger("deepface").setLevel(logging.WARNING)
logging.getLogger("tensorflow").setLevel(logging.WARNING)

class FaceRecognitionTransformer(VideoTransformerBase):
    def __init__(self, config):
        self.config = config
        self.recognized_log = []
        self.frame_counter = 0
        self.last_analysis_time = datetime.now()
        self.analysis_interval = 1e-3  # Analisar a cada 1 ms
        self.lock = threading.Lock()
        
    def get_log(self):
        with self.lock:
            return self.recognized_log.copy()
    
    def analyze_single_face(self, face_crop):
        """Análise de uma única face"""
        try:
            analyzed = DeepFace.analyze(
                img_path=face_crop,
                actions=['age', 'gender'],
                enforce_detection=False,
                detector_backend=self.config['detector_backend'],
                align=self.config['align'],
                silent=self.config['silent'],
                anti_spoofing=self.config['anti_spoofing'],
                expand_percentage=self.config['expand_percentage']
            )
            
            # Garantir que analyzed seja uma lista
            if not isinstance(analyzed, list):
                analyzed = [analyzed]
            
            return analyzed[0] if analyzed else None
            
        except Exception as e:
            error_msg = str(e).lower()
            # Se é erro de spoofing, retornar dados indicando spoofing
            if "spoof" in error_msg or "anti" in error_msg:
                return {
                    'age': 25,
                    'dominant_gender': 'Man',
                    'real_face': False,
                    'spoofing_detected': True,
                    'region': None  # Será preenchido depois
                }
            return None
    
    def recognize_single_face(self, face_crop):
        """Reconhecimento de uma única face"""
        try:
            if not os.path.exists('./celebrity_faces/'):
                return None
                
            recognized = DeepFace.find(
                img_path=face_crop,
                db_path='./celebrity_faces/',
                model_name=self.config['model_face_recognition'],
                enforce_detection=False,
                detector_backend='skip',
                distance_metric=self.config['distance_metric'],
                align=self.config['align'],
                silent=self.config['silent']
            )
            return recognized
        except Exception:
            return None
    
    def analyze_and_recognize(self, frame):
        """Análise e reconhecimento facial melhorado"""
        try:
            # Primeiro, detectar todas as faces usando OpenCV como fallback
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces_opencv = face_cascade.detectMultiScale(gray, 1.1, 4)
            
            # Tentar análise completa com DeepFace primeiro
            analyzed_all = None
            try:
                analyzed_all = DeepFace.analyze(
                    img_path=frame,
                    actions=['age', 'gender'],
                    enforce_detection=False,
                    detector_backend=self.config['detector_backend'],
                    align=self.config['align'],
                    silent=self.config['silent'],
                    anti_spoofing=self.config['anti_spoofing'],
                    expand_percentage=self.config['expand_percentage']
                )
                
                if not isinstance(analyzed_all, list):
                    analyzed_all = [analyzed_all]
                    
            except Exception as e:
                # Se a análise completa falhar, processar face por face
                analyzed_all = []
                
            # Se não conseguiu analisar com DeepFace, processar cada face individualmente
            if not analyzed_all and len(faces_opencv) > 0:
                analyzed_all = []
                for (x, y, w, h) in faces_opencv:
                    # Extrair a face
                    margin = 20
                    x_start = max(0, x - margin)
                    y_start = max(0, y - margin)
                    x_end = min(frame.shape[1], x + w + margin)
                    y_end = min(frame.shape[0], y + h + margin)
                    
                    face_crop = frame[y_start:y_end, x_start:x_end]
                    
                    if face_crop.size > 0:
                        # Analisar esta face específica
                        face_result = self.analyze_single_face(face_crop)
                        if face_result:
                            # Adicionar informações de região
                            face_result['region'] = {'x': x, 'y': y, 'w': w, 'h': h}
                            analyzed_all.append(face_result)
            
            # Agora processar reconhecimento para cada face
            recognized_results = []
            spoofing_results = []
            
            for face_data in analyzed_all:
                # Verificar spoofing
                is_spoofing = False
                if self.config['anti_spoofing']:
                    is_spoofing = (face_data.get("real_face") is False or 
                                 face_data.get("spoofing_detected") is True)
                
                spoofing_results.append(is_spoofing)
                
                # Tentar reconhecimento se não for spoofing
                if not is_spoofing:
                    region = face_data.get('region')
                    if region:
                        x, y, w, h = region['x'], region['y'], region['w'], region['h']
                        
                        # Extrair a face para reconhecimento
                        margin = 20
                        x_start = max(0, x - margin)
                        y_start = max(0, y - margin)
                        x_end = min(frame.shape[1], x + w + margin)
                        y_end = min(frame.shape[0], y + h + margin)
                        
                        face_crop = frame[y_start:y_end, x_start:x_end]
                        
                        if face_crop.size > 0:
                            # Redimensionar para otimizar o reconhecimento
                            face_h, face_w = face_crop.shape[:2]
                            if face_w > 200:
                                scale = 200 / face_w
                                new_w = int(face_w * scale)
                                new_h = int(face_h * scale)
                                face_crop = cv2.resize(face_crop, (new_w, new_h))
                            
                            # Reconhecer esta face específica
                            recognized = self.recognize_single_face(face_crop)
                            recognized_results.append(recognized)
                        else:
                            recognized_results.append(None)
                    else:
                        recognized_results.append(None)
                else:
                    # Se for spoofing, não tentar reconhecer
                    recognized_results.append(None)
            
            return analyzed_all, recognized_results, spoofing_results
            
        except Exception as e:
            # Fallback final: usar OpenCV para detectar faces básicas
            try:
                face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, 1.1, 4)
                
                fallback_results = []
                for (x, y, w, h) in faces:
                    fallback_results.append({
                        'region': {'x': x, 'y': y, 'w': w, 'h': h},
                        'age': 25,
                        'dominant_gender': 'Man',
                        'real_face': True,
                        'spoofing_detected': False
                    })
                
                return fallback_results, [None] * len(fallback_results), [False] * len(fallback_results)
                
            except Exception:
                return [], [], []

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        self.frame_counter += 1
        
        # Controle de frequência de análise
        current_time = datetime.now()
        should_analyze = (current_time - self.last_analysis_time).total_seconds() > self.analysis_interval
        
        if should_analyze:
            analyzed, recognized_results, spoofing_results = self.analyze_and_recognize(img)
            self.last_analysis_time = current_time
        else:
            # Usar resultados anteriores se existirem
            analyzed = getattr(self, 'last_analyzed', [])
            recognized_results = getattr(self, 'last_recognized', [])
            spoofing_results = getattr(self, 'last_spoofing', [])
        
        # Salvar resultados para próximos frames
        if analyzed is not None:
            self.last_analyzed = analyzed
            self.last_recognized = recognized_results
            self.last_spoofing = spoofing_results
        
        # Processar resultados para cada pessoa
        if analyzed and len(analyzed) > 0:
            for i, face in enumerate(analyzed):
                region = face.get('region')
                if not region:
                    continue
                
                x, y, w, h = region['x'], region['y'], region['w'], region['h']
                idade = face.get('age', 0)
                genero = 'Homem' if face.get('dominant_gender') == 'Man' else 'Mulher'
                
                # Verificar spoofing para esta face específica
                is_spoofing = False
                if i < len(spoofing_results):
                    is_spoofing = spoofing_results[i]
                
                # Determinar nome e cor
                if is_spoofing:
                    name = "SPOOFING DETECTADO"
                    color = (0, 0, 255)  # Vermelho
                    erro_status = "Spoofing"
                else:
                    name = "Desconhecido"
                    color = (0, 255, 0)  # Verde
                    erro_status = ""
                    
                    # Tentar reconhecer esta face específica
                    if (i < len(recognized_results) and 
                        recognized_results[i] is not None and 
                        len(recognized_results[i]) > 0 and 
                        not recognized_results[i][0].empty):

                        best_match = recognized_results[i][0].iloc[0]
                        distancia = best_match['distance']
                        if distancia < 0.90:
                            identity_path = best_match['identity']
                            folder_name = os.path.basename(os.path.dirname(identity_path))
                            name = folder_name if folder_name else "Desconhecido"
                            color = (255, 0, 0)  # Azul para reconhecido
                        else:
                            name = "Desconhecido"

    
                
                # Log de reconhecimento
                now = datetime.now()
                dois_min = now - timedelta(minutes=2)
                
                with self.lock:
                    # Verificar duplicatas considerando posição da face para evitar logs duplicados da mesma pessoa
                    face_id = f"{name}_{x//50}_{y//50}"  # ID baseado em nome e posição aproximada
                    duplicate = any(
                        r.get("Face_ID") == face_id and r["Datetime"] >= dois_min 
                        for r in self.recognized_log
                    )
                    
                    if (is_spoofing or name not in ["Desconhecido"]) and not duplicate:
                        self.recognized_log.append({
                            "Nome": name,
                            "Idade": int(idade),
                            "Gênero": genero,
                            "Hora": now.strftime("%H:%M:%S"),
                            "Erro": erro_status,
                            "Datetime": now,
                            "Face_ID": face_id  # Para controle de duplicatas
                        })
                        
                        # Salvar CSV
                        try:
                            df = pd.DataFrame(self.recognized_log[::-1])
                            df_display = df.drop(columns=["Datetime", "Face_ID"])
                            df_display.to_csv("log_reconhecimento.csv", index=False)
                        except Exception:
                            pass
                
                # Desenhar no rosto
                display_text = name
                
                # Caixa de texto maior e mais visível
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.8
                thickness = 2
                (tw, th), baseline = cv2.getTextSize(display_text, font, font_scale, thickness)
                
                # Retângulo da face
                cv2.rectangle(img, (x, y), (x + w, y + h), color, 3)
                
                # Fundo do texto
                cv2.rectangle(img, (x, y - th - 15), (x + tw + 10, y), color, -1)
                
                # Texto principal
                cv2.putText(img, display_text, (x + 5, y - 8), 
                           font, font_scale, (255, 255, 255), thickness)
                
                # Informação adicional
                info_text = f"Idade: {int(idade)} | {genero}"
                cv2.putText(img, info_text, (x + 5, y + h + 25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                
                if self.config['anti_spoofing']:
                    status_text = f"Real: {'Nao' if is_spoofing else 'Sim'}"
                    cv2.putText(img, status_text, (x + 5, y + h + 45),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                
                
        return av.VideoFrame.from_ndarray(img, format="bgr24")

def main():
    st.set_page_config(page_title="Reconhecimento Facial em Tempo Real", layout="wide")
    
    # Configurações na sidebar
    st.sidebar.title("⚙️ Configurações")
    
    config = {
        "detector_backend": st.sidebar.selectbox("Detector Backend",
            ['retinaface','opencv','mtcnn','ssd','dlib','mediapipe','yolov8','centerface','skip'],
            index=0),
        "align": st.sidebar.checkbox("Alinhar rostos", value=True),
        "anti_spoofing": st.sidebar.checkbox("Anti-spoofing", value=False),
        "distance_metric": st.sidebar.selectbox("Métrica de distância",
            ['cosine','euclidean','euclidean_l2'], index=0),
        "model_face_recognition": st.sidebar.selectbox("Modelo de reconhecimento",
            ["Facenet512","VGG-Face","Facenet","OpenFace","DeepFace",
             "DeepID","Dlib","ArcFace","SFace","GhostFaceNet"], index=0),
        "expand_percentage": st.sidebar.slider("Expandir detecção (%)", 0, 100, 0),
        "silent": st.sidebar.checkbox("Modo silencioso", value=True)
    }
    
    # Título principal
    st.title("🧠 Aplicativo de Reconhecimento Facial")
    st.markdown("---")
    
    # Alertas e informações
    if config['anti_spoofing']:
        st.info("🛡️ Anti-spoofing ativado! Cada face é verificada individualmente.")
    
    # Seção de cadastro
    st.subheader("📸 Cadastre um novo usuário")
    uploaded = st.file_uploader("Selecione uma imagem", type=["jpg","jpeg","png"])
    
    if uploaded is not None:
        folder = ''.join(filter(str.isalpha, uploaded.name.split('.')[0]))
        path = f"./celebrity_faces/{folder}/"
        os.makedirs(path, exist_ok=True)
        with open(f"{path}/{uploaded.name}", "wb") as f:
            f.write(uploaded.getbuffer())
        st.success(f"✅ Imagem {uploaded.name} adicionada com sucesso!")
    
    st.markdown("---")
    
    # Layout em colunas
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📹 Webcam ao vivo")
        
        # Configurações WebRTC
        webrtc_ctx = webrtc_streamer(
            key="facial-recognition",
            mode=WebRtcMode.SENDRECV,
            video_transformer_factory=lambda: FaceRecognitionTransformer(config),
            media_stream_constraints={"video": True, "audio": False},
            async_transform=True,
            rtc_configuration={
                "iceServers": [
                    {"urls": ["stun:stun.l.google.com:19302"]},
                    {"urls": ["stun:stun1.l.google.com:19302"]},
                ]
            }
        )
    
    with col2:
        st.subheader("📋 Log de Reconhecimentos")
        log_placeholder = st.empty()
        
        # Atualizar log em tempo real
        if webrtc_ctx.video_transformer:
            recognized_log = webrtc_ctx.video_transformer.get_log()
            if recognized_log:
                df = pd.DataFrame(recognized_log[::-1])
                df_display = df.drop(columns=["Datetime", "Face_ID"])
                log_placeholder.dataframe(df_display, use_container_width=True)
            else:
                log_placeholder.info("Nenhum reconhecimento registrado ainda.")
        
        # Botão para limpar log
        if st.button("🗑️ Limpar Log"):
            if webrtc_ctx.video_transformer:
                with webrtc_ctx.video_transformer.lock:
                    webrtc_ctx.video_transformer.recognized_log.clear()
                st.success("Log limpo com sucesso!")
    
    # Informações adicionais
    st.markdown("---")
    st.subheader("📊 Informações do Sistema")
    
    info_col1, info_col2, info_col3 = st.columns(3)
    
    with info_col1:
        st.metric("Detector", config['detector_backend'])
    
    with info_col2:
        st.metric("Modelo", config['model_face_recognition'])
    
    with info_col3:
        st.metric("Anti-spoofing", "Ativado" if config['anti_spoofing'] else "Desativado")
    
    # Instruções
    st.markdown("---")
    st.subheader("📋 Instruções")
    st.markdown("""
    1. **Permitir acesso à webcam**: Clique em "START" e permita o acesso à sua webcam quando solicitado
    2. **Cadastrar usuários**: Use o upload de imagem acima para adicionar novos rostos ao banco de dados
    3. **Monitorar reconhecimentos**: Veja os resultados em tempo real no log à direita
    4. **Configurar parâmetros**: Ajuste as configurações na barra lateral conforme necessário
   """)
    
    # Rodapé
    st.markdown("---")
    st.markdown("*Sistema de Reconhecimento Facial em Tempo Real*")

if __name__ == "__main__":
    main()
