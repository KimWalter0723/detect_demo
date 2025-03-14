document.getElementById("startCameraButton").addEventListener("click", startCamera);
document.getElementById("startDetectionButton").addEventListener("click", startDetection);

let session = null;  // YOLO 模型
let isDetecting = false;  // 是否正在检测

async function startCamera() {
    const video = document.getElementById("video");
    
    // 访问摄像头
    navigator.mediaDevices.getUserMedia({ video: true })
        .then(stream => {
            video.srcObject = stream;
            document.getElementById("startDetectionButton").disabled = false; // 启用识别按钮
        })
        .catch(err => console.error("无法访问摄像头", err));

    // 预加载 YOLO ONNX 模型
    session = await ort.InferenceSession.create('best.onnx', {
        executionProviders: ['webgl']
    });
}

async function startDetection() {
    if (!session) {
        alert("模型尚未加载，请稍候...");
        return;
    }

    isDetecting = !isDetecting;  // 切换识别状态
    document.getElementById("startDetectionButton").textContent = isDetecting ? "🛑 停止识别" : "🎯 开始识别";

    if (isDetecting) {
        detectObjects();
    }
}

async function detectObjects() {
    if (!isDetecting) return;

    const video = document.getElementById("video");
    const canvas = document.getElementById("canvas");
    const ctx = canvas.getContext("2d");

    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    // 获取图像数据
    const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
    const inputTensor = new ort.Tensor('float32', imageData.data, [1, 3, canvas.height, canvas.width]);

    // 运行 YOLO 推理
    const results = await session.run({ 'input': inputTensor });

    displayResults(results);
    requestAnimationFrame(detectObjects); // 继续检测
}

// 显示检测结果
function displayResults(results) {
    const resultBox = document.getElementById("resultsList");
    resultBox.innerHTML = "";  // 清空旧结果

    results.output.forEach(item => {
        const listItem = document.createElement("li");
        listItem.textContent = `类别: ${item.class}, 置信度: ${(item.confidence * 100).toFixed(2)}%`;
        resultBox.appendChild(listItem);
    });
}
