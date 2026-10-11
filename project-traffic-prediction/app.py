import gradio as gr
import pandas as pd
import numpy as np
import joblib
import os

base_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(base_dir, 'models', 'traffic_volume_rf_model.pkl')
lag_data_path = os.path.join(base_dir, 'models', 'latest_lag_data.pkl')
csv_path = os.path.join(base_dir, 'v_aadt_2565.csv')

if not os.path.exists(model_path):
    model_path = os.path.join(base_dir, 'traffic_volume_rf_model.pkl')
    lag_data_path = os.path.join(base_dir, 'latest_lag_data.pkl')

try:
    model = joblib.load(model_path)
    latest_lag_data = joblib.load(lag_data_path)
except Exception as e:
    model = None
    latest_lag_data = {}

province_map = {
    'กบ': 'กระบี่', 'กท': 'กรุงเทพมหานคร', 'กทม': 'กรุงเทพมหานคร', 'กจ': 'กาญจนบุรี', 
    'กส': 'กาฬสินธุ์', 'กพ': 'กำแพงเพชร', 'ขก': 'ขอนแก่น', 'จบ': 'จันทบุรี', 
    'ฉช': 'ฉะเชิงเทรา', 'ชบ': 'ชลบุรี', 'ชน': 'ชัยนาท', 'ชย': 'ชัยภูมิ', 
    'ชพ': 'ชุมพร', 'ชร': 'เชียงราย', 'ชม': 'เชียงใหม่', 'ตง': 'ตรัง', 
    'ตร': 'ตราด', 'ตก': 'ตาก', 'นย': 'นครนายก', 'นฐ': 'นครปฐม', 
    'นพ': 'นครพนม', 'นม': 'นครราชสีมา', 'นศ': 'นครศรีธรรมราช', 'นว': 'นครสวรรค์', 
    'นบ': 'นนทบุรี', 'นธ': 'นราธิวาส', 'นน': 'น่าน', 'บก': 'บึงกาฬ', 
    'บร': 'บุรีรัมย์', 'ปท': 'ปทุมธานี', 'ปข': 'ประจวบคีรีขันธ์', 'ปจ': 'ปราจีนบุรี', 
    'ปน': 'ปัตตานี', 'พย': 'พะเยา', 'อย': 'พระนครศรีอยุธยา', 'พง': 'พังงา', 
    'พท': 'พัทลุง', 'พจ': 'พิจิตร', 'พล': 'พิษณุโลก', 'พบ': 'เพชรบุรี', 
    'พช': 'เพชรบูรณ์', 'พร': 'แพร่', 'ภก': 'ภูเก็ต', 'มค': 'มหาสารคาม', 
    'มห': 'มุกดาหาร', 'มส': 'แม่ฮ่องสอน', 'ยส': 'ยโสธร', 'ยล': 'ยะลา', 
    'รอ': 'ร้อยเอ็ด', 'รน': 'ระนอง', 'รย': 'ระยอง', 'รบ': 'ราชบุรี', 
    'ลบ': 'ลพบุรี', 'ลป': 'ลำปาง', 'ลพ': 'ลำพูน', 'ลย': 'เลย', 
    'ศก': 'ศรีสะเกษ', 'สน': 'สกลนคร', 'สข': 'สงขลา', 'สต': 'สตูล', 
    'สป': 'สมุทรปราการ', 'สส': 'สมุทรสงคราม', 'สค': 'สมุทรสาคร', 'สก': 'สระแก้ว', 
    'สบ': 'สระบุรี', 'สห': 'สิงห์บุรี', 'สท': 'สุโขทัย', 'สพ': 'สุพรรณบุรี', 
    'สฎ': 'สุราษฎร์ธานี', 'สร': 'สุรินทร์', 'นค': 'หนองคาย', 'นภ': 'หนองบัวลำภู', 
    'อท': 'อ่างทอง', 'อจ': 'อำนาจเจริญ', 'อด': 'อุดรธานี', 'อต': 'อุตรดิตถ์', 
    'อน': 'อุทัยธานี', 'อบ': 'อุบลราชธานี'
}

link_mapping = {}
try:
    if os.path.exists(csv_path):
        try:
            df_info = pd.read_csv(csv_path, encoding='utf-8')
        except UnicodeDecodeError:
            df_info = pd.read_csv(csv_path, encoding='cp874')
            
        df_unique = df_info.drop_duplicates(subset=['link_id'])
        for _, row in df_unique.iterrows():
            l_id = row['link_id']
            road = str(row['road_code']) if pd.notna(row['road_code']) else ""
            

            prov_abbr = str(row['province_code']).strip().replace('.', '') if pd.notna(row['province_code']) else ""
            prov_full = province_map.get(prov_abbr, prov_abbr)
            
            desc = f"{road} จ.{prov_full}".strip()
            
            if desc and desc != "จ.":
                link_mapping[l_id] = f"{l_id} - {desc}"
            else:
                link_mapping[l_id] = str(l_id)
except Exception as e:
    print(f"Warning: Could not load CSV for mapping - {e}")

available_links_raw = sorted(list(latest_lag_data.keys())) if latest_lag_data else [98]
dropdown_choices = [link_mapping.get(l, str(l)) for l in available_links_raw]
default_choice = dropdown_choices[0] if dropdown_choices else None


def predict_traffic(budget_year: int, link_id_display: str) -> str:
    if model is None:
        return "<div class='output-value' style='font-size:1.5rem; color:red;'>ไม่พบไฟล์ Model</div>"
        
    try:
        link_id_str = str(link_id_display).split(" - ")[0].strip()
        link_id = int(link_id_str)
        
        if link_id not in latest_lag_data:
             return "<div class='output-value' style='font-size:1.5rem; color:red;'>ไม่พบข้อมูลอดีตสำหรับรหัสสายทางนี้</div>"
             
        aadt_lag1 = latest_lag_data[link_id]
        input_data = pd.DataFrame({
            'budget_year': [budget_year], 
            'link_id': [link_id],
            'aadt_lag1': [aadt_lag1]
        })
        
        prediction_log = model.predict(input_data)
        prediction = np.expm1(prediction_log)
        
        predicted_aadt = max(0, round(prediction[0]))
        
        return f"<div class='output-value'>{predicted_aadt:,.0f} <span class='output-unit'>คัน/วัน</span></div>"
        
    except Exception as e:
        return f"<div class='output-value' style='font-size:1.5rem; color:red;'>Error: {str(e)}</div>"

custom_css = """
@import url('https://fonts.googleapis.com/css2?family=Prompt:wght@300;400;500;600;700&display=swap');

body, * { font-family: 'Prompt', sans-serif !important; }
body { background-color: #f1f5f9; }

.header-bg { 
    background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%); 
    color: white; 
    padding: 35px 30px; 
    border-radius: 24px; 
    margin-bottom: 25px; 
    box-shadow: 0 10px 30px rgba(59, 130, 246, 0.25); 
    text-align: center;
}
.header-bg h1 { color: white !important; margin-top: 0; font-weight: 700; letter-spacing: 0.5px; }
.header-bg p { color: #e0e7ff; font-size: 1.15rem; margin-bottom: 0; font-weight: 300; }

.dashboard-card { 
    background: #ffffff !important; 
    border-radius: 24px !important; 
    padding: 25px !important; 
    box-shadow: 0 8px 24px rgba(149, 157, 165, 0.1) !important; 
    border: none !important; 
}

.output-value { 
    font-size: 4rem; 
    font-weight: 700; 
    background: linear-gradient(to right, #2563eb, #0d9488);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-align: center; 
    margin: 15px 0; 
    line-height: 1.2;
}
.output-unit {
    font-size: 1.5rem;
    font-weight: 500;
    -webkit-text-fill-color: #64748b;
}

.kpi-box { 
    background: #f8fafc; 
    border-radius: 16px; 
    padding: 15px; 
    text-align: center; 
    border: 1px solid #e2e8f0; 
    transition: all 0.3s ease;
}
.kpi-box:hover {
    transform: translateY(-4px);
    box-shadow: 0 10px 20px -5px rgba(0, 0, 0, 0.08);
    border-color: #cbd5e1;
}
.kpi-box h4 { color: #64748b; margin: 0 0 6px 0; font-size: 0.9rem; text-transform: uppercase; letter-spacing: 1px; font-weight: 600; }
.kpi-box h2 { color: #0f172a; margin: 0; font-size: 1.8rem; font-weight: 700; }

button.primary {
    background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%) !important;
    border: none !important;
    box-shadow: 0 4px 12px rgba(37, 99, 235, 0.3) !important;
    font-weight: 600 !important;
    border-radius: 12px !important;
    transition: all 0.2s;
}
button.primary:hover {
    background: linear-gradient(135deg, #1d4ed8 0%, #1e3a8a 100%) !important;
    transform: scale(1.01);
}
"""

custom_theme = gr.themes.Soft(font=gr.themes.GoogleFont("Prompt"))

with gr.Blocks(title="Traffic Analytics Prediction") as app:
    
    gr.HTML(
        """
        <div class='header-bg'>
            <h1>Traffic Analytics Prediction</h1>
            <p>ระบบพยากรณ์ปริมาณการจราจรเฉลี่ยรายวัน (AADT) ด้วยแบบจำลอง Machine Learning</p>
        </div>
        """
    )
    
    with gr.Row():
        with gr.Column(scale=1, elem_classes="dashboard-card"):
            gr.Markdown("### กำหนดพารามิเตอร์")
            budget_year_input = gr.Slider(minimum=2555, maximum=2575, step=1, value=2566, label="ปีงบประมาณ (Budget Year)")
            
            link_id_input = gr.Dropdown(
                choices=dropdown_choices, 
                value=default_choice, 
                label="สายทาง (Link ID & Road Code)"
            )
            
            predict_button = gr.Button("ประมวลผลการทำนาย", variant="primary", size="lg")
            gr.Markdown("<br><p style='color:#64748b; font-size:0.9rem; text-align:center;'>เลือกสายทางที่ต้องการแล้วกดปุ่มประมวลผล</p>")

        with gr.Column(scale=2):
            with gr.Row(elem_classes="dashboard-card"):
                with gr.Column():
                    gr.Markdown("### ปริมาณจราจรที่คาดการณ์")
                    output_display = gr.HTML("<div class='output-value'>- <span class='output-unit'>คัน/วัน</span></div>")
            
            with gr.Row(elem_classes="dashboard-card"):
                with gr.Column():
                    gr.Markdown("### ประสิทธิภาพแบบจำลอง (Model Metrics)")
                    with gr.Row():
                        gr.HTML("<div class='kpi-box'><h4>R-Squared</h4><h2>0.9158</h2></div>")
                        gr.HTML("<div class='kpi-box'><h4>MAE Error</h4><h2>825 <span style='font-size:0.9rem; color:#64748b;'>คัน/วัน</span></h2></div>")
            
            with gr.Row():
                with gr.Column(scale=1, elem_classes="dashboard-card"):
                    gr.Markdown("### ตัวอย่างชุดข้อมูล")
                    
                    ex_list = []
                    if len(dropdown_choices) > 0: ex_list.append([2566, dropdown_choices[0]])
                    if len(dropdown_choices) > 10: ex_list.append([2567, dropdown_choices[10]])
                    if len(dropdown_choices) > 20: ex_list.append([2570, dropdown_choices[20]])
                    
                    gr.Examples(
                        examples=ex_list,
                        inputs=[budget_year_input, link_id_input],
                        label=""
                    )
                
                with gr.Column(scale=1, elem_classes="dashboard-card"):
                    gr.Markdown("### ข้อมูลเชิงประจักษ์")
                    gr.Markdown(
                        """
                        * **การแปรผลลัพธ์:** โมเดลจะแสดงตัวเลขคาดการณ์รถที่จะวิ่งผ่านถนนเส้นนั้นๆ ต่อวัน 
                        * **ประโยชน์:** ใช้เพื่อวางแผนงบประมาณซ่อมบำรุงและขยายช่องจราจรล่วงหน้า
                        * **ข้อจำกัด:** อาจมีความคลาดเคลื่อนในสายทางที่มีการเติบโตแบบก้าวกระโดดผิดปกติ
                        """
                    )

    predict_button.click(
        fn=predict_traffic,
        inputs=[budget_year_input, link_id_input],
        outputs=output_display
    )

if __name__ == "__main__":
    host = "0.0.0.0"
    port = 7860
    app.launch(
        server_name=host,
        server_port=port,
        theme=custom_theme,
        css=custom_css
    )