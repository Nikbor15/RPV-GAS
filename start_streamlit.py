from app import run_streamlit_multi

if __name__ == "__main__":
    run_streamlit_multi(
        data_dir="data",          # pasta onde você vai colocar os 11 xlsx
        weights_csv=None,         # opcional: caminho p/ CSV com colunas: ticker,weight
        use_embedded_weights=True # usa os pesos embutidos abaixo (pode trocar p/ False)
    )
