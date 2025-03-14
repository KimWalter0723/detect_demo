document.getElementById("startCameraButton").addEventListener("click", startCamera);
document.getElementById("startDetectionButton").addEventListener("click", startDetection);
document.getElementById("stopDetectionButton").addEventListener("click", stopDetection);

let session = null;  
let isDetecting = false;  

async function startCamera() {
    const video = document.getElementById("video");

    navigator.mediaDevices.getUserMedia({ video: true })
        .then(stream => {
            video.srcObject = stream;
            document.getElementById("startDetectionButton").disabled = false; 
        })
        .catch(err => console.error("无法访问摄像头", err));

    console.log("📢 正在加载 YOLO 模型...");

    try {
        session = await ort.InferenceSession.create('/yolo_model.onnx', {
            executionProviders: ['wasm'] // 兼容性更强，适用于 Edge
        });
        console.log("✅ YOLO 模型加载完成！");
        alert("模型已加载，可以开始检测！");
    } catch (err) {
        console.error("❌ YOLO 模型加载失败：", err);
        alert("模型加载失败，请检查控制台错误信息！");
    }
}


async function startDetection() {
    if (!session) {
        alert("模型尚未加载，请稍候...");
        return;
    }

    isDetecting = true;
    document.getElementById("startDetectionButton").disabled = true;
    document.getElementById("stopDetectionButton").disabled = false;
    detectObjects();
}

function stopDetection() {
    isDetecting = false;
    document.getElementById("startDetectionButton").disabled = false;
    document.getElementById("stopDetectionButton").disabled = true;
}

async function detectObjects() {
    if (!isDetecting) return;

    const video = document.getElementById("video");
    const canvas = document.getElementById("canvas");
    const ctx = canvas.getContext("2d");

    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
    const inputTensor = new ort.Tensor('float32', imageData.data, [1, 3, canvas.height, canvas.width]);

    const results = await session.run({ 'input': inputTensor });

    displayResults(results);
    requestAnimationFrame(detectObjects);
}

// 显示识别结果
function displayResults(results) {
    const resultBox = document.getElementById("resultsList");
    resultBox.innerHTML = "";  

    results.output.forEach(item => {
        const listItem = document.createElement("li");
        listItem.textContent = `类别: ${item.class}, 置信度: ${(item.confidence * 100).toFixed(2)}%`;
        resultBox.appendChild(listItem);
    });
}
