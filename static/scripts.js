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

