# 🚀 FIRE-EYE: Safety Equipment Detector for Space Stations

A YOLOv8-based object detection system designed to identify critical fire safety items onboard simulated space stations using Duality AI's Falcon-generated dataset.

## 👨‍🚀 Project Summary

FIRE-EYE ensures that essential emergency tools like Fire Extinguishers, Tool Boxes, and Oxygen Tanks are always available and detectable — preventing safety risks in space environments. 

This submission includes:
- 📄 Final Report (`report.pdf`)
- 🧠 Trained YOLOv8 Model (`best.pt`)
- 🎮 Streamlit App (`streamlit_app.py`)
- 📦 Requirements (`requirements.txt`)
- 🎨 UI Assets (`/assets`)
- 📊 Training Logs (`/runs/detect/train3`)

---

## 🔍 Detection Classes
- FireExtinguisher
- ToolBox
- OxygenTank

---

## 💡 Dataset
We used the official **HackByte_Dataset** provided in the AlgoVerse challenge. No external datasets were used.

---

## 📈 Final Performance
| Metric | Value |
|--------|-------|
| mAP@0.5 | **0.939** |
| mAP@0.5:0.95 | **0.866** |
| FireExtinguisher Precision | 1.000 |
| ToolBox Precision | 0.982 |
| OxygenTank Precision | 0.948 |

---

## ⚙️ How to Run

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
