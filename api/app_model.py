from fastapi import FastAPI, HTTPException, Query, Request
import uvicorn
import os
import pickle
import pandas as pd
import sqlite3
from typing import Optional


def init_db():
    df = pd.read_csv('./data/Advertising.csv')
    conn = sqlite3.connect('data/advertising.db')
    df.to_sql('Advertising', conn, if_exists='replace', index=False)
    conn.close()

init_db()

# Cargar el modelo
with open('data/advertising_model.pkl', 'rb') as f:
    model = pickle.load(f)


#os.chdir(os.path.dirname(__file__))

app = FastAPI()

@app.get("/")
async def read_root():
    return {"message": "Bienvenido a mi API del modelo advertising"}

# 1. Endpoint de predicción
@app.get("/predict")
async def predict(tv: float, radio: float, newspaper: float):
    try:
        data_inversion = {'TV': tv, 'radio': radio, 'newspaper': newspaper}
        input_df = pd.DataFrame([data_inversion])
        prediction = model.predict(input_df)
        return {"Prediction": round(prediction[0], 2)}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# 2. Endpoint de ingesta de datos
@app.post("/v1/ingest")
async def ingest(request: Request):
    data = await request.json()

    tv = data.get('tv')
    radio = data.get('radio')
    newspaper = data.get('newspaper')
    sales = data.get('sales')

    if tv is None or radio is None or newspaper is None or sales is None:
        raise HTTPException(status_code=400, detail="Faltan datos, los valores de TV, radio, newspaper y sales son necesarios")

    # Insertar los datos en la base de datos SQLite
    try:
        conn = sqlite3.connect('data/advertising.db')
        c = conn.cursor()
        c.execute("INSERT INTO advertising_data (tv, radio, newspaper, sales) VALUES (?, ?, ?, ?)", (tv, radio, newspaper, sales))
        conn.commit()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        conn.close()

    return {"message": "Datos recibidos y almacenados correctamente"}


# 3. Endpoint de reentrenamiento del modelo
@app.post("/v1/retrain")
def retrain():
    try:
        conn = sqlite3.connect('data/advertising.db')
        df = pd.read_sql_query("SELECT * FROM advertising_data", conn)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        conn.close()

    X = df[['tv', 'radio', 'newspaper']]
    y = df['sales']

    # Reentrenar el modelo existente
    model.fit(X, y)

    # Guardar el modelo reentrenado
    with open('data/advertising_model.pkl', 'wb') as f:
        pickle.dump(model, f)

    return {"message": "Modelo reentrenado con éxito"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

