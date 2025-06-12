# 🧠 Aplicativo de Reconhecimento Facial em Tempo Real com Streamlit

## 🧾 Funcionalidades

- 📸 Captura em tempo real da webcam
- 😎 Detecção de:
  - Emoção (feliz, triste, bravo, etc.)
  - Gênero
  - Idade estimada
  - Raça
- 🧠 Reconhecimento facial
- 🔐 Modo anti-spoofing (opcional)
- 🛠️ Escolha entre diversos backends de detecção
- 📁 Reconhecimento com base em rostos salvos na pasta `celebrity_faces/`

---

## 🗂️ Estrutura do Projeto


```
FACE_RECOGNITION_STREAMLIT/
│
├── celebrity_faces/               # Pasta com imagens de referência
│   ├── Hyago/
│   ├── Jennifer Lawrence/
│   └── ...
│
├── app.py                         # Arquivo principal da aplicação
├── requirements.txt               # Dependências
├── .gitignore
├── README.md
└── images/                        # Imagens usadas no README
    ├── analyze_example.png
    └── recognition_example.png
```

---

## ▶️ Como usar

### 1. Clone este repositório

```bash
git clone https://github.com/HyAgOsK/Rec_face_app.git
cd Rec_face_app
```

### 3. Crie um ambiente virtual com Python 3.8

[Versões do python](https://www.python.org/downloads/release/python-380/)

```bash

python -m venv venv
./venv/Script/activate
```

Se você estiver usando o VS CODE, clique no canto inferior esquerdo, em azul, conecte com o executavel para ambiente virtual
com python venv, usando o executável que você baixou na web.

### 2. Instale os requisitos

```bash
pip install -r requirements.txt
```

> 💡 Recomenda-se usar um ambiente virtual (`venv` python=3.9).

### 3. Execute o Streamlit

```bash
streamlit run app.py
```

---

## 📦 Requisitos principais

> Veja todos os pacotes no arquivo [`requirements.txt`](./requirements.txt)

---

## 📥 Adicionando novas faces ao banco

Adicione imagens de pessoas que você deseja reconhecer dentro da pasta pelo proprio aplicativo:

```
celebrity_faces/NomeDaPessoa/
```

O DeepFace usará estas imagens como base de comparação.

---

## 🧪 Exemplos de uso

- Verifique sua emoção em tempo real pela webcam
- Veja sua idade estimada
- Detecte seu gênero e raça
- Verifique se você é reconhecido(a) pelo banco de dados

---

Desenvolvido com 💡 por [Hyago]