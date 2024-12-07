from flask import Flask, render_template, request, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from spacy.training.example import Example
from spacy.training import offsets_to_biluo_tags
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
import telebot
import pandas as pd
import os
import spacy
import pickle
import pyodbc
import csv
import random
import json
import re
import joblib
import threading
import asyncio

app = Flask(__name__)

# Đường dẫn tệp dữ liệu và nơi lưu model
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "train-model-data.csv")
MODEL_PATH = os.path.join(BASE_DIR, "food_recognition3_model")

nlp = spacy.load("food_recognition3_model")
# Load TF-IDF data
with open('artifacts/tfidf_data.pkl', 'rb') as f:
    tfidf_data = joblib.load(f)

# Kết nối cơ sở dữ liệu
conn = pyodbc.connect(
    'DRIVER={SQL Server};'
    'SERVER=TRONG-NGHIA\\SERVER0;'
    'DATABASE=QLMA;'
    'Trusted_Connection=yes;'
)
cursor = conn.cursor()

# Lấy dữ liệu từ database
query = """exec sp_GetDataDish"""
data = pd.read_sql_query(query, conn)

# Lấy lại vectorizer, TF-IDF matrix và dữ liệu gốc
vectorizer = tfidf_data['vectorizer']
tfidf_matrix_db = tfidf_data['tfidf_matrix']
data = tfidf_data['data']

# Hàm loại bỏ thực thể trùng lặp
def remove_duplicates(entities):
    return list(set(entities))  # Loại bỏ trùng lặp bằng cách chuyển thành set rồi chuyển lại list

# Hàm kiểm tra và loại bỏ thực thể chồng lấn
def remove_overlaps(entities):
    sorted_entities = sorted(entities, key=lambda x: (x[0], x[1]))  # Sắp xếp thực thể theo vị trí
    non_overlapping = []
    for ent in sorted_entities:
        if not non_overlapping or ent[0] >= non_overlapping[-1][1]:  # Kiểm tra chồng lấn
            non_overlapping.append(ent)
    return non_overlapping

def convert_csv_to_spacy_format(file_path):
    train_data = []
    with open(file_path, newline='', encoding='utf-8-sig') as csvfile:
        reader = csv.DictReader(csvfile, delimiter='\t')  # Sử dụng tab làm phân tách
        for row in reader:
            text = row["Sentence"]
            try:
                entities = eval(row["Entities"])  # Chuyển chuỗi thành list
                entities_list = [(ent['start'], ent['end'], ent['label']) for ent in entities]
                entities_list = remove_duplicates(entities_list)  # Loại bỏ thực thể trùng lặp
                entities_list = remove_overlaps(entities_list)  # Loại bỏ chồng lấn
                train_data.append((text, {"entities": entities_list}))
            except Exception as e:
                print(f"Error processing row: {row}")
                print(f"Error: {e}")
    return train_data

def extract_positions(sentence, nguyenlieu_list, cachchebien_list):
    positions = []
    
    # Xử lý nguyên liệu
    for entity in nguyenlieu_list:
        start_index = sentence.lower().find(entity.lower())
        if start_index != -1:
            end_index = start_index + len(entity)
            positions.append({"start": start_index, "end": end_index, "label": "NGUYEN_LIEU"})
    
    # Xử lý cách chế biến
    for technique in cachchebien_list:
        start_index = sentence.lower().find(technique.lower())
        if start_index != -1:
            end_index = start_index + len(technique)
            positions.append({"start": start_index, "end": end_index, "label": "CACH_CHE_BIEN"})
    
    return positions      

def normalize_text(text):
    # Viết thường
    text = text.lower()
    # Loại bỏ khoảng trắng dư ở đầu/đuôi và chuyển khoảng trắng dư giữa các từ thành một khoảng trắng
    text = re.sub(r'\s+', ' ', text.strip())
    return text

def extract_labels(user_input):
    user_input = user_input.lower()
    
    # Tải mô hình spaCy
    nlp = spacy.load("food_recognition3_model")
    doc = nlp(user_input)
    
    # Tìm kiếm các thực thể với nhãn tương ứng
    nguyenLieu = [ent.text for ent in doc.ents if ent.label_ == "NGUYEN_LIEU"]
    cachCheBien = [ent.text for ent in doc.ents if ent.label_ == "CACH_CHE_BIEN"]
    
    return nguyenLieu, cachCheBien

def suggest_dish(user_input):
    global vectorizer, tfidf_matrix_db, data
    
    with open('artifacts/tfidf_data.pkl', 'rb') as f:
        tfidf_data = joblib.load(f)
        
    nguyenLieu, cachCheBien = extract_labels(user_input)
    query = ", ".join(nguyenLieu + cachCheBien)
    vectorizer = tfidf_data['vectorizer']
    tfidf_matrix_db = tfidf_data['tfidf_matrix']
    data = tfidf_data['data']
       
    if query:
        # Vectorize input và tính cosine similarity
        tfidf_vector_user = vectorizer.transform([query])
        similarity_scores = cosine_similarity(tfidf_vector_user, tfidf_matrix_db)
        top_indices = similarity_scores.argsort()[0][-5:][::-1]  # Top 5 món ăn
        
        # Trả về danh sách món ăn
        suggestions = []
        for idx in top_indices:
            dish = data.iloc[idx]
            suggestions.append((dish['MonAn'], dish['CachCheBien'], dish['NguyenLieu']))
        return suggestions
    return None

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_message = update.message.text
    print(f"User input: {user_message}")
    
    # Gợi ý món ăn
    suggestions = suggest_dish(user_message)
    
    if suggestions:
        response = "Top 5 món ăn gợi ý:\n"
        for idx, (dish, recipe, ingredients) in enumerate(suggestions, 1):
            response += f"{idx}. {dish}\n   Cách chế biến: {recipe}\n   Nguyên liệu: {ingredients}\n"
    else:
        response = "Xin lỗi, tôi không nhận diện được nguyên liệu hoặc cách chế biến từ tin nhắn của bạn."
    
    # Phản hồi lại người dùng
    await update.message.reply_text(response)

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Xin chào! Hãy gửi tin nhắn để tôi gợi ý món ăn dựa trên nguyên liệu và cách chế biến của bạn.")
    
def reload_tfidf_data():
    # Đọc lại tfidf_data.pkl để cập nhật các biến toàn cục
    with open('artifacts/tfidf_data.pkl', 'rb') as f:
        tfidf_data = joblib.load(f)
        
    global vectorizer, tfidf_matrix_db, data
    vectorizer = tfidf_data['vectorizer']
    tfidf_matrix_db = tfidf_data['tfidf_matrix']
    data = tfidf_data['data']

def run_bot():
    global bot_running
    bot_running = True
    TELEGRAM_API_TOKEN = "7595143737:AAGp-DATccxWD0-RZyoS6A1ru3E0kEoVmas"
    application = Application.builder().token(TELEGRAM_API_TOKEN).build()
    
    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    
    reload_tfidf_data()
    
    # Tạo event loop mới cho thread con này
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    # Chạy bot Telegram với event loop asyncio
    loop.run_until_complete(application.run_polling())

def stop_bot():
    global bot_running
    if bot_running:
        # Dừng vòng lặp bot
        bot_running = False
        print("Bot đã được tắt.")
        # Nếu có cơ chế tắt thread của bot, dùng nó ở đây
        # Ví dụ: application.stop() hoặc một cách tắt khác
        if bot_thread:
            bot_thread.join()  # Chờ bot thread kết thúc

# Biến lưu trữ trạng thái bot
bot_thread = None
bot_running = False

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/partials/<string:page>')
def load_partial(page):
    if page == "Setting":
        return render_template('partials/Setting.html')
    if page == "Bot-Telegram":
        return render_template('partials/Bot-Telegram.html')
    if page == "Train-Model":
        return render_template('partials/Train-Model.html')
    if page == "ManagerDish":
        query = """
            SELECT 
                ma.ID AS MonAnID, 
                ma.Ten AS MonAnTen, 
                cc.Ten AS CachCheBien, 
                STRING_AGG(nl.Ten, ', ') AS NguyenLieu
            FROM tb_MonAn ma
            LEFT JOIN tb_Cachchebien cc ON ma.CachchebienID = cc.ID
            LEFT JOIN tb_ChitietMonan ctma ON ma.ID = ctma.MonAnID
            LEFT JOIN tb_NguyenLieu nl ON ctma.NguyenLieuID = nl.ID
            GROUP BY ma.ID, ma.Ten, cc.Ten
        """
        cursor.execute(query)
        results = cursor.fetchall()
        dishes = [
            {"MonAnID": row.MonAnID, "MonAnTen": row.MonAnTen, "CachCheBien": row.CachCheBien, "NguyenLieu": row.NguyenLieu}
            for row in results
        ]
        return render_template('partials/ManagerDish.html', dishes=dishes)
    return jsonify({"error": "Page not found"}), 404

@app.route('/add_mon_an', methods=['POST'])
def add_mon_an():
    data = request.json
    mon_an = data.get('monAn')
    nguyen_lieu = data.get('nguyenLieu')
    cach_che_bien = data.get('cachCheBien')

    if not mon_an or not nguyen_lieu or not cach_che_bien:
        return jsonify(success=False, message="Dữ liệu không hợp lệ!")

    # Kết nối cơ sở dữ liệu
    conn = pyodbc.connect(
        'DRIVER={SQL Server};SERVER=TRONG-NGHIA\\SERVER0;DATABASE=QLMA;Trusted_Connection=yes;'
    )
    cursor = conn.cursor()

    try:
        # Kiểm tra và thêm cách chế biến
        cursor.execute("SELECT ID FROM tb_Cachchebien WHERE Ten = ?", cach_che_bien)
        row = cursor.fetchone()
        if row:
            cach_che_bien_id = row[0]
        else:
            cursor.execute("INSERT INTO tb_Cachchebien (Ten) VALUES (?)", cach_che_bien)
            conn.commit()
            cach_che_bien_id = cursor.execute("SELECT @@IDENTITY").fetchone()[0]

        # Kiểm tra và thêm món ăn
        cursor.execute("SELECT ID FROM tb_MonAn WHERE Ten = ?", mon_an)
        row = cursor.fetchone()
        if row:
            return jsonify(success=False, message="Món ăn đã tồn tại!")

        cursor.execute("INSERT INTO tb_MonAn (Ten, CachchebienID) VALUES (?, ?)", mon_an, cach_che_bien_id)
        conn.commit()
        mon_an_id = cursor.execute("SELECT @@IDENTITY").fetchone()[0]

        # Hàm chuẩn hóa nguyên liệu
        def normalize_ingredient(ingredient):
            # Loại bỏ khoảng trắng dư ở đầu/đuôi, chuyển khoảng trắng dư giữa các từ thành một khoảng trắng
            ingredient = " ".join(ingredient.split())
            # Viết hoa chữ cái đầu tiên của cụm
            return ingredient.capitalize()

        # Xử lý nguyên liệu
        nguyen_lieu_list = [
            normalize_ingredient(ng)  # Chuẩn hóa từng cụm nguyên liệu
            for ng in nguyen_lieu.split(',')  # Tách các nguyên liệu dựa trên dấu phẩy
            if ng.strip()  # Loại bỏ nguyên liệu rỗng (nếu có)
        ]

        # Tiếp tục với quy trình xử lý như trước
        for nguyen_lieu_name in nguyen_lieu_list:
            # Kiểm tra nếu nguyên liệu đã tồn tại trong database
            cursor.execute("SELECT ID FROM tb_NguyenLieu WHERE LTRIM(RTRIM(Ten)) = ?", nguyen_lieu_name)
            row = cursor.fetchone()
            if row:
                nguyen_lieu_id = row[0]
            else:
                # Thêm nguyên liệu mới vào tb_NguyenLieu
                cursor.execute("INSERT INTO tb_NguyenLieu (Ten) VALUES (?)", nguyen_lieu_name)
                conn.commit()
                nguyen_lieu_id = cursor.execute("SELECT @@IDENTITY").fetchone()[0]

            # Thêm chi tiết món ăn vào tb_ChitietMonan
            cursor.execute("INSERT INTO tb_ChitietMonan (MonAnID, NguyenLieuID) VALUES (?, ?)", mon_an_id, nguyen_lieu_id)

        conn.commit()
        # Lấy dữ liệu từ database
        query = """exec sp_GetDataDish"""
        data = pd.read_sql_query(query, conn)
        return jsonify(success=True)

    except Exception as e:
        conn.rollback()
        return jsonify(success=False, message=str(e))
    finally:
        conn.close()

@app.route('/update-tfidf', methods=['POST'])
def update_tfidf():
    try:
        global vectorizer, tfidf_matrix_db, data
        # Kết nối đến cơ sở dữ liệu
        conn = pyodbc.connect(
            'DRIVER={SQL Server};'
            'SERVER=TRONG-NGHIA\\SERVER0;'
            'DATABASE=QLMA;'
            'Trusted_Connection=yes;'
        )

        # Lấy dữ liệu mới từ database
        query = """exec sp_GetDataDish"""
        data = pd.read_sql_query(query, conn)

        # Xử lý TF-IDF
        data['Text'] = data['NguyenLieu'] + ', ' + data['CachCheBien']
        vectorizer = TfidfVectorizer(lowercase=True)
        tfidf_matrix_db = vectorizer.fit_transform(data['Text'])

        # Lưu dữ liệu mới vào file `tfidf_data.pkl`
        tfidf_path = os.path.join(os.getcwd(), "artifacts", "tfidf_data.pkl")
        with open(tfidf_path, 'wb') as f:
            pickle.dump({'vectorizer': vectorizer, 'tfidf_matrix': tfidf_matrix_db, 'data': data}, f)
        
        vectorizer = tfidf_data['vectorizer']
        tfidf_matrix_db = tfidf_data['tfidf_matrix']
        data = tfidf_data['data']

        conn.close()
        return jsonify(success=True)

    except Exception as e:
        print("Lỗi khi cập nhật TF-IDF:", str(e))
        return jsonify(success=False, message=str(e))

@app.route('/train-model', methods=['POST'])
def train_model():
    try:
        # Load dữ liệu từ file CSV
        train_data = convert_csv_to_spacy_format(DATA_PATH)

        # Tạo model spaCy
        nlp = spacy.blank("vi")
        ner = nlp.add_pipe("ner", last=True)
        ner.add_label("NGUYEN_LIEU")
        ner.add_label("CACH_CHE_BIEN")

        optimizer = nlp.begin_training()

        # Huấn luyện model
        for itn in range(10):
            print(f"Epoch {itn + 1} bắt đầu...")
            random.shuffle(train_data)
            for text, annotations in train_data:
                doc = nlp.make_doc(text)
                example = Example.from_dict(doc, annotations)
                nlp.update([example], drop=0.5)
            print(f"Epoch {itn + 1} hoàn thành.")

        # Lưu model
        nlp.to_disk(MODEL_PATH)
        return jsonify({"success": True, "message": "Huấn luyện hoàn tất. Model đã được lưu!"})
    except Exception as e:
        return jsonify({"success": False, "message": f"Lỗi: {str(e)}"})

@app.route('/update-data-train', methods=['POST'])
def update_data_train():
    try:
        # Kết nối đến cơ sở dữ liệu
        conn = pyodbc.connect(
            'DRIVER={SQL Server};'
            'SERVER=TRONG-NGHIA\\SERVER0;'
            'DATABASE=QLMA;'
            'Trusted_Connection=yes;'
        )
        
        nguyenlieu_query = "SELECT n.Ten FROM tb_NguyenLieu AS n"
        nguyenlieu_df = pd.read_sql(nguyenlieu_query, conn)

        cachchebien_query = "SELECT c.Ten FROM tb_Cachchebien AS c"
        cachchebien_df = pd.read_sql(cachchebien_query, conn)

        nguyenlieu_list = [normalize_text(item) for item in nguyenlieu_df['Ten']]
        cachchebien_list = [normalize_text(item) for item in cachchebien_df['Ten']]
        
        conn.close()

        file_path = "data/raw-data-train.txt" 
        with open(file_path, "r", encoding="utf-8") as file:
            sentences = [normalize_text(line) for line in file.readlines()]

        df = pd.DataFrame(sentences, columns=["Sentence"])

        df["Entities"] = df["Sentence"].apply(
            lambda sentence: extract_positions(sentence, nguyenlieu_list, cachchebien_list)
        )

        df["Entities"] = df["Entities"].apply(lambda x: json.dumps(x, ensure_ascii=False))

        # Lưu DataFrame thành file CSV với mã hóa UTF-8
        output_file = "data/train-model-data.csv"
        df.to_csv(output_file, index=False, sep="\t", encoding="utf-8-sig")
        
        return jsonify({"success": True, "message": "Dữ liệu đã được cập nhật!"})
    except Exception as e:
        return jsonify({"success": False, "message": f"Lỗi: {str(e)}"})
              
@app.route('/toggle-bot', methods=['POST'])
def toggle_bot():
    global bot_running, bot_thread
    if bot_running:
        # Dừng bot
        stop_bot()
        return jsonify({'bot_running': False})
    else:
        # Bật bot
        bot_thread = threading.Thread(target=run_bot)
        bot_thread.start()
        return jsonify({'bot_running': True})

if __name__ == '__main__':
    app.run(debug=True)
