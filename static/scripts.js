function loadPartial(page) {
    const panel = document.getElementById("content-panel");
    panel.innerHTML = "<p>Loading...</p>"; // Hiển thị loading khi đang tải

    fetch(`/partials/${page}`)
        .then(response => {
            if (!response.ok) {
                throw new Error("Page not found");
            }
            return response.text();
        })
        .then(html => {
            panel.innerHTML = html; // Chèn nội dung động vào panel
        })
        .catch(error => {
            panel.innerHTML = `<p>Error: ${error.message}</p>`;
        });
}

let selectedRow = null;

function selectRow(row) {
    const cells = row.getElementsByTagName("td");
    document.getElementById("monAn").value = cells[0].textContent;
    document.getElementById("nguyenLieu").value = cells[1].textContent;
    document.getElementById("cachCheBien").value = cells[2].textContent;
    selectedRow = row;
}

function addMonAn() {
    const monAn = document.getElementById("monAn").value.trim();
    const nguyenLieu = document.getElementById("nguyenLieu").value.trim();
    const cachCheBien = document.getElementById("cachCheBien").value.trim();

    if (!monAn || !nguyenLieu || !cachCheBien) {
        alert("Vui lòng điền đầy đủ thông tin!");
        return;
    }

    const payload = {
        monAn,
        nguyenLieu,
        cachCheBien,
    };

    fetch("/add_mon_an", {
        method: "POST",
        headers: {
            "Content-Type": "application/json",
        },
        body: JSON.stringify(payload),
    })
        .then((response) => response.json())
        .then((data) => {
            if (data.success) {
                alert("Thêm món ăn thành công!");
                location.reload();
            } else {
                alert("Có lỗi xảy ra: " + data.message);
            }
        })
        .catch((error) => {
            console.error("Error:", error);
            alert("Có lỗi xảy ra trong quá trình xử lý.");
        });
}

// Hàm cập nhật TF-IDF
function updateTfidf() {
    // Hiển thị thông báo đang cập nhật
    alert("Đang cập nhật TF-IDF...");

    // Gửi yêu cầu đến server để cập nhật dữ liệu
    fetch('/update-tfidf', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
    })
        .then(response => {
            if (!response.ok) {
                throw new Error('Có lỗi xảy ra trong quá trình cập nhật!');
            }
            return response.json();
        })
        .then(data => {
            if (data.success) {
                alert("Cập nhật TF-IDF thành công!");
            } else {
                alert("Cập nhật thất bại: " + data.message);
            }
        })
        .catch(error => {
            console.error("Lỗi:", error);
            alert("Có lỗi xảy ra khi kết nối đến server!");
        });
}

function trainModel() {
    // Hiển thị thông báo khi quá trình huấn luyện bắt đầu
    alert("Huấn luyện model... Vui lòng đợi.");

    fetch('/train-model', {
        method: 'POST',
    })
        .then(response => response.json())  // Nhận dữ liệu JSON từ server
        .then(data => {
            // Hiển thị thông báo thành công hoặc lỗi bằng alert
            if (data.success) {
                alert(`Thành công: ${data.message}`);  // Thông báo thành công
            } else {
                alert(`Lỗi: ${data.message}`);  // Thông báo lỗi
            }
        })
        .catch(err => {
            console.error(err);
            alert('Có lỗi xảy ra khi kết nối với server!');  // Thông báo lỗi khi không kết nối được server
        });
}

// viet function update-data-train de update du lieu train
function updateDataTrain() {
    alert("Đang cập nhật dữ liệu train... Vui lòng đợi.");

    fetch('/update-data-train', {
        method: 'POST',
    })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                alert(`Thành công: ${data.message}`);
            } else {
                alert(`Lỗi: ${data.message}`);
            }
        })
        .catch(err => {
            console.error(err);
            alert('Có lỗi xảy ra khi kết nối với server!');
        });
}

function toggleBot() {
    fetch("/toggle-bot", {
        method: "POST",
    })
    .then(response => response.json())
    .then(data => {
        const button = document.getElementById("toggle-bot-btn");
        if (data.bot_running) {
            button.textContent = "Running";
            button.className = "btn running";
            alert("Bot đã được bật và đang chạy!");
        } else {
            button.textContent = "Run Bot";
            button.className = "btn stopped";
            alert("Bot đã tắt.");
        }
    })
    .catch(error => {
        console.error("Error:", error);
        alert("Có lỗi xảy ra khi thực hiện yêu cầu.");
    });
}

function addUserInputData() {
    fetch('/add-user-input-data', {
        method: 'POST',
    })
    .then(response => {
        if (response.ok) {
            alert('Cập nhật dữ liệu thành công!');
        } else {
            alert('Có lỗi xảy ra khi cập nhật dữ liệu.');
        }
    })
    .catch(error => {
        console.error('Error:', error);
        alert('Có lỗi xảy ra khi gửi yêu cầu.');
    });
}