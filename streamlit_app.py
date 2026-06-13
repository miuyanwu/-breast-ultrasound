# Hugging Face Spaces 入口 — streamlit run app.py
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
exec(open(os.path.join(os.path.dirname(__file__), 'app', 'main.py'), encoding='utf-8').read())
