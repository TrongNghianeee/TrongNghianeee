from flask import Flask, render_template, request, jsonify
import pyodbc

app = Flask(__name__)

# Kết nối cơ sở dữ liệu
conn = pyodbc.connect(
    'DRIVER={SQL Server};'
    'SERVER=TRONG-NGHIA\\SERVER0;'
    'DATABASE=QLMA;'
    'Trusted_Connection=yes;'
)
cursor = conn.cursor()


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/partials/<string:page>')
def load_partial(page):
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
        return jsonify(success=True)

    except Exception as e:
        conn.rollback()
        return jsonify(success=False, message=str(e))
    finally:
        conn.close()

if __name__ == '__main__':
    app.run(debug=True)
