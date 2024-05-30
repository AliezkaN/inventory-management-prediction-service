from flask import Flask, request, jsonify
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
app = Flask(__name__)
import warnings
warnings.filterwarnings("ignore")


@app.route('/retail-flow-prediction-manager/predict', methods=['POST'])
def predict():
    data = request.json["stats"]
    print(data)
    sales_data = {}
    for month, sales in data.items():
        for item in sales:
            if item['productId'] not in sales_data:
                sales_data[item['productId']] = {
                    'productName': item['productName'],
                    'sales': []
                }
            sales_data[item['productId']]['sales'].append(item['quantitySold'])

    # Prepare time series data
    time_series_data = []
    for product_id, product_data in sales_data.items():
        product_name = product_data['productName']
        sales = product_data['sales']
        time_series_data.append((product_id, pd.Series(sales, name='Quantity', index=pd.date_range(start=month, periods=len(sales), freq='ME'))))

    # SARIMA model parameters
    order = (1, 1, 1)  # example order parameters (p, d, q)
    seasonal_order = (0, 1, 1, 12)  # example seasonal order parameters (P, D, Q, S)

    forecast_results = {}
    for product_id, sales_series in time_series_data:
        sarima_model = SARIMAX(sales_series, order=order, seasonal_order=seasonal_order, enforce_stationarity=False, enforce_invertibility=False)
        sarima_model_fit = sarima_model.fit(disp=False)
        forecast = sarima_model_fit.forecast(steps=1)
        forecast_results[product_id] = forecast[0]

    predictions = {}
    for product_id, forecast_quantity in forecast_results.items():
        predictions[product_id] = forecast_quantity
        print(f"Product: {product_id}, Forecasted Quantity: {forecast_quantity:.2f}")

    return jsonify({"prediction": predictions})


if __name__ == '__main__':
    app.run(debug=True)
