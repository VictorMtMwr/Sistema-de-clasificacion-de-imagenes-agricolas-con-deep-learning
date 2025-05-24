const imagenInput = document.getElementById('imagenInput');
const imagenPreview = document.getElementById('imagenPreview');
const resultadoDiv = document.getElementById('resultado');
const fileNameSpan = document.getElementById('file-name');

imagenInput.addEventListener('change', () => {
    const file = imagenInput.files[0];
    if (file) {
        imagenPreview.src = URL.createObjectURL(file);
        imagenPreview.style.display = "block";
        fileNameSpan.textContent = file.name;
    } else {
        imagenPreview.style.display = "none";
        fileNameSpan.textContent = "";
    }
});

document.querySelector('.custom-label').addEventListener('click', () => {
    imagenInput.click();
});

function enviarImagen() {
    const file = imagenInput.files[0];
    if (!file) {
        resultadoDiv.textContent = "Por favor selecciona una imagen.";
        return;
    }

    resultadoDiv.textContent = "Procesando...";
    const formData = new FormData();
    formData.append("file", file);

    fetch("http://127.0.0.1:5000/predict", {
        method: "POST",
        body: formData
    })
    .then(res => res.json())
    .then(data => {
        if (data.type && data.probability !== undefined) {
            resultadoDiv.innerHTML = `La hoja es de tipo <b>${data.type}</b> con una probabilidad de <b>${(data.probability*100).toFixed(2)}%</b>.`;
        } else if (data.error) {
            resultadoDiv.textContent = "Error: " + data.error;
        } else {
            resultadoDiv.textContent = "Respuesta inesperada del servidor.";
        }
    })
    .catch(err => {
        resultadoDiv.textContent = "Error en la predicción";
        console.error(err);
    });
}