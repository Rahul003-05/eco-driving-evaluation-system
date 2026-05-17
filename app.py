from flask import Flask, render_template, request, session
import pickle
import numpy as np

app = Flask(__name__)
app.secret_key = "eco_bus_super_secret_key"

model = pickle.load(open('models/best_model.pkl', 'rb'))
scaler = pickle.load(open('models/scaler.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/graphs')
def graphs():
    user_data = session.get('last_features', [0]*12)
    user_result = session.get('last_result', 'Unknown')
    return render_template('graphs.html', user_data=user_data, user_result=user_result)

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        scenario = request.form.get('scenario', 'entering') 
        
        features = [
            float(request.form['max_speed']),
            float(request.form['std_nonzero_speed']),
            float(request.form['max_brake_pedal']),
            float(request.form['avg_brake_pedal']),
            float(request.form['brake_gt_2']),
            float(request.form['max_accel']),
            float(request.form['mean_pos_accel']),
            float(request.form['mean_neg_accel']),
            float(request.form['mean_jerk']),
            float(request.form['std_abs_jerk']),
            float(request.form['decel_time_pct']),
            float(request.form['cpk_value'])
        ]
        
        session['last_features'] = features
        
        final_features = scaler.transform([np.array(features)])
        prediction = model.predict(final_features)
        
        scenario_text = "Entering Stop" if scenario == 'entering' else "Leaving Stop"
        
        if prediction[0] == 1:
            session['last_result'] = 'Eco-Driving'
            result_text = f"Eco-Driving Detected ({scenario_text})"
            sub_text = "Efficient driving pattern. Lower energy used, regenerative braking optimized."
            color = "#2ecc71" 
            icon = "✅"
            bg_image = "https://images.unsplash.com/photo-1544620347-c4fd4a3d5957?q=80&w=2000" 
            
            # Advanced Insights
            why_result = "Your acceleration and braking inputs are within the optimal threshold for energy conservation. Low jerk values indicate a smooth trajectory."
            quick_tip = "Maintain this smooth braking pattern to maximize regenerative battery charging."
            recommendation = "Continue to anticipate stops early. Keep mean jerk below 0.1 m/s³ to ensure maximum passenger comfort."
        else:
            session['last_result'] = 'Non Eco-Driving'
            result_text = f"Non Eco-Driving Detected ({scenario_text})"
            sub_text = "Aggressive approach. High energy consumption and passenger discomfort noted."
            color = "#e74c3c" 
            icon = "⚠️"
            bg_image = "https://images.unsplash.com/photo-1600320844746-8e1467406ce6?q=80&w=2000"
            
            # Advanced Insights
            why_result = "High values in maximum acceleration and heavy brake pedal usage triggered the aggressive driving flag, leading to battery drain."
            quick_tip = "Release the accelerator earlier when approaching the stop to utilize natural deceleration."
            recommendation = "Focus on reducing your 'Brake GT 2m/s³' percentage. Hard braking wastes kinetic energy that could be recovered by the EV battery."
            
        return render_template('result.html', 
                               prediction_text=result_text, 
                               sub_text=sub_text,
                               color=color,
                               icon=icon,
                               bg_image=bg_image,
                               why_result=why_result,
                               quick_tip=quick_tip,
                               recommendation=recommendation)

if __name__ == "__main__":
    app.run(debug=True, port=5000)