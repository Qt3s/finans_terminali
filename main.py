import streamlit as st
import pandas as pd
import numpy as np

st.title("Alper'in Finans Terminali 🚀")

st.write("Kripto ve Hisse Senedi Analiz Platformu çok yakında burada!")

# Örnek bir grafik
chart_data = pd.DataFrame(
    np.random.randn(20, 3),
    columns=['BTC', 'ETH', 'SOL'])

st.line_chart(chart_data)
