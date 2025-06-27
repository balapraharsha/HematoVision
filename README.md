# HematoVision 🔬🧠

**HematoVision** is a deep learning-based web application for **automated blood cell classification** using a Convolutional Neural Network (CNN) and transfer learning. It features a user-friendly interface built with **Flask**, allowing users to upload microscopic images and receive instant predictions.

---

## 📁 Project Structure

```
HematoVision
├── PROJECT EXECUTABLE FILES/
│   ├── app.py                      # Flask web application
│   ├── model.py                    # CNN model loading & prediction
│   ├── blood_cell.h5               # Trained deep learning model
│   ├── static/                     # CSS & images
│   ├── templates/                  # HTML UI
│   ├── README.md
├── PROJECT INITIALIZATION AND PLANNING/
│   ├── Define Problem Statement.pdf
│   ├── Project Planning.pdf
│   ├── Proposed Solution.pdf
│   ├── README.md
├── PROJECT DOCUMENTATION.pdf       # Combined project report (root level)
├── requirements.txt                # Python dependencies (root level)
├── LICENSE                         # License file (root level)
├── README.md                       # Main README (root level)
```

---

## 🚀 Features

- 🔍 Classifies 4 types of blood cells (e.g., Neutrophil, Eosinophil, Monocyte, Lymphocyte)
- ⚡ Fast predictions with pre-trained CNN (Transfer Learning)
- 🌐 Flask-based web interface with drag-and-drop upload
- 🎨 Clean UI with custom styles and illustrations
- 🧠 Suitable for remote diagnosis and education

---

## 🛠️ Tech Stack

- Python
- TensorFlow / Keras
- Flask
- HTML, CSS, JavaScript (Jinja2)
- Git & GitHub
- Git LFS (for large `.h5` model file)

---

## 👨‍👩‍👧‍👦 Team

| Role          | Name                                |
|---------------|-------------------------------------|
| **Team Leader** | Mannepalli Bala Praharsha           |
| **Member**      | Marripudi Varshini                  |
| **Member**      | Pesala Balasaraswathimeghana        |
| **Member**      | Reddy Venkata Sai Sidhardha         |

---

## 📝 How to Run

1. Clone the repo:
   ```bash
   git clone https://github.com/balapraharsha/HematoVision.git

2. Navigate into the project folder:
   ```bash
    cd "PROJECT EXECUTABLE FILES"

3. (Optional) Install Git LFS and pull the large model file
   ``` bash
    git lfs install
    git lfs pull

4. Install required Python dependencies
   ```bash
    pip install -r requirements.txt

5. Run the Flask application
   ```bash
    python app.py


---

<p align="center">📄 License
This project is for academic and educational use only.
© 2025 HematoVision Team. All rights reserved.</p>

<p align="center"> Made with ❤️ by the HematoVision Team </p> 
