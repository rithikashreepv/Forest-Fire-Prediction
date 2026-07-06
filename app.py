
import gradio as gr
import pandas as pd
import numpy as np
import joblib


model = joblib.load("forest_fire_random_forest.pkl")
preprocessor = joblib.load("preprocessor.pkl")




def predict_fire(
    X, Y,
    month, day,
    FFMC, DMC, DC, ISI,
    temp, RH, wind, rain
):

    # Create DataFrame
    input_data = pd.DataFrame({
        "X": [X],
        "Y": [Y],
        "month": [month],
        "day": [day],
        "FFMC": [FFMC],
        "DMC": [DMC],
        "DC": [DC],
        "ISI": [ISI],
        "temp": [temp],
        "RH": [RH],
        "wind": [wind],
        "rain": [rain]
    })

    # Apply preprocessing
    input_processed = preprocessor.transform(input_data)

    # Predict
    prediction = model.predict(input_processed)

    # Convert from log scale
    prediction = np.expm1(prediction)

    return f"""
### 🔥 Prediction Complete

**Estimated Burned Area:** **{prediction[0]:.2f} hectares**

> **Note:** This prediction is generated using a tuned Random Forest Regressor trained on the UCI Forest Fires dataset. The result is an estimate and should not be interpreted as an exact real-world burned area.
"""



examples = [
    [7, 5, "mar", "fri", 86.2, 26.2, 94.3, 5.1, 8.2, 51, 6.7, 0.0],
    [7, 4, "oct", "tue", 90.6, 35.4, 669.1, 6.7, 18.0, 33, 0.9, 0.0]
]




with gr.Blocks(theme=gr.themes.Soft(), title="Forest Fire Predictor") as demo:

    gr.Markdown("""
# 🌲 Forest Fire Burned Area Prediction

Predict the estimated burned area of a forest fire using meteorological conditions and fire weather indices.

### 🔍 Model Information

- 🌳 **Algorithm:** Random Forest Regressor
- ⚙️ **Hyperparameter Optimization:** GridSearchCV (5-Fold Cross Validation)
- 🔄 **Categorical Encoding:** One-Hot Encoding
- 📈 **Target Transformation:** Log1p
- 📊 **Dataset:** UCI Forest Fires Dataset
""")

    with gr.Row():

        with gr.Column():

            x = gr.Number(label="📍 X Coordinate")
            y = gr.Number(label="📍 Y Coordinate")

            month = gr.Dropdown(
                choices=[
                    "jan","feb","mar","apr","may","jun",
                    "jul","aug","sep","oct","nov","dec"
                ],
                label="📅 Month"
            )

            day = gr.Dropdown(
                choices=[
                    "mon","tue","wed","thu",
                    "fri","sat","sun"
                ],
                label="📅 Day"
            )

            ffmc = gr.Number(
                label="🔥 FFMC (Fine Fuel Moisture Code)"
            )

            dmc = gr.Number(
                label="🌿 DMC (Duff Moisture Code)"
            )

            dc = gr.Number(
                label="🌳 DC (Drought Code)"
            )

            isi = gr.Number(
                label="💨 ISI (Initial Spread Index)"
            )

        with gr.Column():

            temp = gr.Number(
                label="🌡️ Temperature (°C)"
            )

            rh = gr.Number(
                label="💧 Relative Humidity (%)"
            )

            wind = gr.Number(
                label="🌬️ Wind Speed (km/h)"
            )

            rain = gr.Number(
                label="🌧️ Rain (mm)"
            )

            output = gr.Markdown()

            predict_btn = gr.Button(
                "🔥 Predict Burned Area",
                variant="primary",
                size="lg"
            )

            clear_btn = gr.ClearButton()

    gr.Examples(
        examples=examples,
        inputs=[
            x, y,
            month, day,
            ffmc, dmc, dc, isi,
            temp, rh, wind, rain
        ]
    )

    predict_btn.click(
        fn=predict_fire,
        inputs=[
            x, y,
            month, day,
            ffmc, dmc, dc, isi,
            temp, rh, wind, rain
        ],
        outputs=output
    )

   

demo.launch()
